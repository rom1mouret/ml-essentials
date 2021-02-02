package algorithms

import (
  "time"
  "log"
  "math"
  "math/rand"
  "github.com/rom1mouret/ml-essentials/utils"
  "github.com/rom1mouret/ml-essentials/dataframe"
  "github.com/rom1mouret/ml-essentials/preprocessing"
  "gonum.org/v1/gonum/floats"
  "gonum.org/v1/gonum/blas/blas64"
  "gonum.org/v1/gonum/mat"
)

// LinRegTrainParams contain hyper-parameters and other training options.
// Epochs, LR and BatchSize will be set to default values if they are left to 0.
type LinRegTrainParams struct {
  // Number of times the entire dataset iterated over.
  // The dataset is shuffled before each epoch.
  Epochs      int
  // Learning rate per row (not per batch) used to update the weights.
  LR          float64
  // Size of the batch across which the gradients will be accumulated.
  BatchSize   int
  // Weights *= (1 - decay) after each iteration. It plays a regularization role
  WeightDecay float64
  // Gradient momentum. Gradients are initialized with PrevGradients * Momentum
  Momentum    float64
  // Allows the algorithm to make copies of the data to minimize cache-misses.
  LowMemory   bool
  // Prints out information like training time and loss value.
  Verbose     bool
}

// LinearRegressor is a model that predicts a numerical target with a linear
// combination of the input features.
// LinearRegressor is serializable with json.
type LinearRegressor struct {
  // json serializable
  TargetScaler     *preprocessing.Scaler
  Weights          []float64
  Features         []string
}

func NewLinearRegressor() *LinearRegressor {
  result := new(LinearRegressor)
  return result
}

// Fit trains a linear regressor on a dataframe that contains both the features
// and the numerical target, identified by the "targetColumn" argument.
// As of now, it always returns nil but it could return an error in future
// versions.
// You need not shuffle the dataset prior to calling this function.
func (reg *LinearRegressor) Fit(df *dataframe.DataFrame, targetColumn string,
                                params LinRegTrainParams) error {
  // just to save some memory if we perform a copy below
  floatsAndBools := df.FloatHeader().And(df.BoolHeader())
  df = df.ColumnView(floatsAndBools.NameList()...)

  // copy the target to another column before scaling it in-place
  start := time.Now()
  df = df.DetachedView(targetColumn)
  df.Rename(targetColumn, "_target")
  opt := preprocessing.ScalerOptions{Centering: true, Scaling: true}
  targetDF := df.ColumnView("_target")
  reg.TargetScaler = preprocessing.NewScaler(opt)
  reg.TargetScaler.Fit(targetDF)
  reg.TargetScaler.TransformInplace(df)
  if params.Verbose {
    log.Printf("training+running Target Scaler took: %s", time.Since(start))
  }
  // training features and their corresponding weights
  reg.Features = floatsAndBools.Except(targetColumn).NameList()
  reg.Weights = make([]float64, len(reg.Features))
  for i := range reg.Weights {
    reg.Weights[i] = rand.NormFloat64()
  }
  HeInitializer := math.Sqrt(2.0/float64(len(reg.Weights)))
  floats.Scale(HeInitializer, reg.Weights)

  // gonum vectors
  gradients := mat.NewVecDense(len(reg.Features), nil)
  weights := mat.NewVecDense(len(reg.Weights), reg.Weights)
  var prevGradients *mat.VecDense
  if params.Momentum > 0 {
    prevGradients = mat.NewVecDense(len(reg.Weights), nil)
  }
  // pre-allocation
  diff := make([]float64, params.BatchSize)

  // training loop
  batching := dataframe.NewDense64Batching(reg.Features)
  for _, epoch := range utils.MakeRange(1, params.Epochs+1, 1) {
    start := time.Now()
    var batches []*dataframe.DataFrame
    if params.LowMemory {
      batches = df.ShuffleView().SplitView(params.BatchSize)
    } else {
      // Copy() sometimes makes DenseMatrix() faster
      batches = df.ShuffleView().Copy().SplitView(params.BatchSize)
    }
    absErr := 0.0
    for _, batch := range batches {
      batchSize := batch.NumRows()
      diffVec := mat.NewVecDense(batchSize, diff[:batchSize])
      y := batch.Floats("_target").VecDense()
      rows := batching.DenseMatrix(batch)
      // diff = rows x weights - y  [dim: batch size]
      diffVec.MulVec(rows, weights)
      diffVec.SubVec(diffVec, y)
      if params.Verbose {
        absErr += blas64.Asum(diffVec.RawVector())
      }
      // gradients = SUM(MulElem(batch, diff), axis=0), i.e. transpose(batch) * diff
      gradients.MulVec(rows.T(), diffVec)
      if params.Momentum > 0 {
        gradients.AddScaledVec(gradients, params.Momentum, prevGradients)
        prevGradients.CopyVec(gradients)
      }
      // weight update
      weights.AddScaledVec(weights, -params.LR, gradients)
      // weight decay
      if params.WeightDecay > 0 {
        weights.ScaleVec(1-params.WeightDecay, weights)
      }
    }
    if params.Verbose {
      mae := absErr / float64(df.NumRows())
      log.Printf("training epoch %d took: %s (scaled MAE: %f)",
                 epoch, time.Since(start), mae)
    }
  }
  return nil
}

// Predict predicts the target and stores the result in a new dataframe in a new
// float column identified by "resultColumn" argument.
// The new dataframe is a view on the original dataframe, so they share most
// data.
// As of now, it always returns nil alongside the result dataframe, but it could
// return an error in future versions.
// If resultColumn already exists, the previous data will be overwritten but it
// will not change the data in the original dataframe.
func (reg *LinearRegressor) Predict(df *dataframe.DataFrame, resultColumn string) (*dataframe.DataFrame, error) {
  df = df.ResetIndexView() // makes batching.DenseMatrix faster

  // simple heuristic to maximize the batch size without blowing up the memory
  batchSize := 4194304 / len(reg.Weights)
  if batchSize < 32 {
    batchSize = 32
  }
  // pre-allocation
  weights := mat.NewVecDense(len(reg.Weights), reg.Weights)
  pred := make([]float64, df.NumRows())

  // prediction
  batching := dataframe.NewDense64Batching(reg.Features)
  for i, batch := range df.SplitView(batchSize) {
    rows := batching.DenseMatrix(batch)
    offset := i * batchSize
    yData := pred[offset:offset+batch.NumRows()]
    yVec := mat.NewVecDense(len(yData), yData)
    yVec.MulVec(rows, weights)
  }

  // write the result in the dataframe
  result := df.View()
  result.OverwriteFloats64("_target", pred)
  reg.TargetScaler.InverseTransformInplace(result)
  result.Rename("_target", resultColumn)

  return result, nil
}

// Here is how to train with a float iterator and without Gonum matrices:
// gradients := make([]float64, len(reg.Weights))
// ite := dataframe.NewFloat64Iterator(df, reg.Features)
// for _, epoch := range utils.MakeRange(1, params.Epochs+1, 1) {
//   start := time.Now()
//   trainErr := 0.0
//   n := 0
//   for _, batch := range df.ShuffleView(-1).SplitView(params.BatchSize) {
//     n += batch.NumRows()
//     y := df.Floats("_target")
//     ite.Reset(batch, false)
//     // reset gradients
//     floats.Scale(params.Momentum, gradients)
//     // accumulate gradients
//     for row, i, _ := ite.NextRow(); row != nil; row, i, _ = ite.NextRow() {
//       pred := floats.Dot(row, reg.Weights)
//       err := pred - y.Get(i)
//       trainErr += math.Abs(err)
//       floats.AddScaled(gradients, err, row)  // grad += row * err
//     }
//     // optional gradient clipping
//     if params.Clipping > 0.0001 {
//       lowerBound := -params.Clipping
//       for k, v := range gradients {
//         if v > params.Clipping {
//           gradients[k] = params.Clipping
//         } else if v < lowerBound {
//           gradients[k] = lowerBound
//         }
//       }
//     }
//     // update weights :  w += -lr * gradients
//     floats.AddScaled(reg.Weights, -params.LR, gradients)
//     if params.WeightDecay > 0 {
//       floats.Scale(1 - params.WeightDecay, reg.Weights)
//     }
//   }
//   if params.Verbose {
//     mae := trainErr / float64(n)
//     log.Printf("training epoch %d took: %s (MAE: %f)", epoch, time.Since(start), mae)
//   }
// }
