package preprocessing

import (
  "math"
  "github.com/rom1mouret/ml-essentials/utils"
  "github.com/rom1mouret/ml-essentials/dataframe"
)

type ScalerOptions struct {
  // if true, Centering and Scaling are ignored
  MinMax    bool
  // with mean
  Centering bool
  // with std
  Scaling   bool
}

// Scaler is a json-serializable structure that lets you center and scale
// float features to avoid your models to be biased towards certain features
// or face initialization issues.
type Scaler struct {
  Shift    map[string]float64
  Scale    map[string]float64
  options  ScalerOptions
}

// NewScaler allocates a new Scaler
func NewScaler(opt ScalerOptions) *Scaler {
  scaler := new(Scaler)
  scaler.options = opt
  return scaler
}

type fittingResult struct {
  shift *float64
  scale *float64
}

func workerFits(df *dataframe.DataFrame, opt ScalerOptions, q utils.StringQ) {
  for col := q.Next(); len(col) > 0; col = q.Next() {
    access := df.Floats(col)
    result := fittingResult{}
    defer q.Notify(utils.ProcessedJob{Key: col, Result: &result})
    // min max scaling
    if opt.MinMax {
      min := math.Inf(1)
      max := math.Inf(-1)
      for i := 0; i < access.Size(); i++ {
        val := access.Get(i)
        if val > max {
          max = val
        }
        if val < min {
          min = val
        }
      }
      shift := -min
      scale := max - min
      result.shift = &shift
      result.scale = &scale
    } else if opt.Scaling || opt.Centering {
      // TODO: bessell correction?
      mean := 0.0
      n := 0
      for i := 0; i < access.Size(); i++ {
        val := access.Get(i)
        if !math.IsNaN(val) {
          mean += val
          n++
        }
      }
      mean /= float64(n)
      if opt.Centering {
        shift := -mean
        result.shift = &shift
      }
      if opt.Scaling {
        squaresum := 0.0
        for i := 0; i < access.Size(); i++ {
          val := access.Get(i)
          if !math.IsNaN(val) {
            squaresum += (val - mean) * (val - mean)
          }
        }
        if squaresum == 0 {
          squaresum += 1
          n++
        }
        scale := math.Sqrt(float64(n) / squaresum)
        result.scale = &scale
      }
    }
  }
}

// Fit implements PreprocTraining and InplaceTransform interfaces.
func (scaler *Scaler) Fit(df *dataframe.DataFrame) error {
  columns := df.FloatHeader().NameList()
  q := df.CreateColumnQueue(columns)
  for i := 0; i < q.Workers; i++ {
    go workerFits(df, scaler.options, q)
  }
  scaler.Shift = make(map[string]float64)
  scaler.Scale = make(map[string]float64)
  for _, job := range q.Results() {
    result := job.Result.(*fittingResult)
    if result.scale != nil {
      scaler.Scale[job.Key] = *result.scale
    }
    if result.shift != nil {
      scaler.Shift[job.Key] = *result.shift
    }
  }
  return nil
}
func (scaler *Scaler) workerTransforms(df *dataframe.DataFrame, q utils.StringQ) {
  for col := q.Next(); len(col) > 0; col = q.Next() {
    defer q.Notify(utils.ProcessedJob{Key: col})
    var ok bool
    var scale float64
    shift := scaler.Shift[col]  // 0 by default
    if scale, ok = scaler.Scale[col]; !ok {
      scale = 1  // default scale
    }
    access := df.Floats(col)
    for i := 0; i < access.Size(); i++ {
      access.Set(i, (access.Get(i) + shift) * scale)
    }
  }
}

// TransformInplace implements InplaceTransform interface.
func (scaler *Scaler) TransformInplace(df *dataframe.DataFrame) error {
  // divide into column groups
  var m map[string]float64
  if len(scaler.Scale) > len(scaler.Shift) {
    m = scaler.Scale
  } else {
    m = scaler.Shift
  }
  if len(m) == 0 {
    return nil
  }
  q := df.CreateColumnQueue(scaler.TransformedColumns())
  defer q.Wait()
  for i := 0; i < q.Workers; i++ {
    go scaler.workerTransforms(df, q)
  }
  return nil
}

// InverseTransformInplace implements InverseInplaceTransform interface.
func (scaler *Scaler) InverseTransformInplace(df *dataframe.DataFrame) error {
  var reverse Scaler
  reverse.Shift = make(map[string]float64)
  reverse.Scale = make(map[string]float64)
  for col, shift := range scaler.Shift {
    if scale, ok := scaler.Scale[col]; ok {
      reverse.Shift[col] = -shift * scale  // it's going to be divided by scale
    } else {
      reverse.Shift[col] = -shift
    }
  }
  for col, scale := range scaler.Scale {
    reverse.Scale[col] = 1 / scale
  }
  return reverse.TransformInplace(df)
}

// TransformedColumns implements PreprocTraining interface and InplaceTransform.
func (scaler *Scaler) TransformedColumns() []string {
  if len(scaler.Shift) > 0 {
    return mapKeys(scaler.Shift)
  } else {
    return mapKeys(scaler.Scale)
  }
}

func mapKeys(set map[string]float64) []string {
  result := make([]string, len(set))
  i := 0
  for key := range set {
    result[i] = key
    i++
  }
  return result
}
