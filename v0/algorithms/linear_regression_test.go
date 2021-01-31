package algorithms

import (
  "testing"
  "math/rand"
  "fmt"
  "github.com/rom1mouret/ml-essentials/v0/utils"
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
  "github.com/rom1mouret/ml-essentials/v0/preprocessing"
)

func Test1LinearRegression(t *testing.T) {
  b := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  for i := 0; i < 5000; i++ {
    x1 := 4 * rand.NormFloat64()
    x2 := x1 + 20
    // target = 1
    b.AddStrings("str", "plus1")
    b.AddFloats("col", x1)
    b.AddFloats("target", 1)
    // target = 2
    b.AddStrings("str", "plus2")
    b.AddFloats("col", x1)
    b.AddFloats("target", 2)
    // target = 2
    b.AddStrings("str", "plus1")
    b.AddFloats("col", x2)
    b.AddFloats("target", 2)
    // target = 3
    b.AddStrings("str", "plus2")
    b.AddFloats("col", x2)
    b.AddFloats("target", 3)
  }
  df := b.ToDataFrame()
  // preprocessor
  preproc := preprocessing.NewAutoPreprocessor(preprocessing.AutoPreprocOptions{
      Imputing: true, Scaling: true, Exclude: []string{"target"}})
  original := df
  df = preproc.FitTransform(df)
  // training
  model := NewLinearRegressor()
  model.Fit(df, "target", LinRegTrainParams{
    Epochs: 50, LR: 0.001, BatchSize: 64, Momentum: 0.3, Verbose: true})
  // prediction
  df, _ = preproc.TransformView(original)
  df, err := model.Predict(df, "y_pred")
  if err != nil {
    panic(err.Error())
  }
  // evaluation
  mae := MAE(df.Floats("target"), df.Floats("y_pred"))
  fmt.Println("mae", mae)
  utils.AssertTrue("mae", mae < 0.16, t)
}
