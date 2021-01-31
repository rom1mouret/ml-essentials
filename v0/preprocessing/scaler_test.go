package preprocessing

import (
  "testing"
  "math"
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
  u "github.com/rom1mouret/ml-essentials/v0/utils"
)

func TestMinMaxScaling(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddFloats("col", 1.0, 1.0, math.NaN(), 2.0, 2.0).ToDataFrame()
  imputer := NewScaler(ScalerOptions{MinMax: true})
  imputer.Fit(df)
  df = df.IndexView([]int{0, 1, 3, 4})
  err := imputer.TransformInplace(df)
  if err != nil {
    panic(err.Error())
  }
  vals := df.Floats("col")
  u.AssertFloatEquals("vals[0]", vals.Get(0), 0.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 0.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 1.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 1.0, t)

  imputer.InverseTransformInplace(df)
  u.AssertFloatEquals("vals[0]", vals.Get(0), 1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 1.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 2.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 2.0, t)
}

func TestCenteringAlone(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddFloats("col", math.NaN(), 1.0, 1.0, 2.0, 2.0).ToDataFrame()
  imputer := NewScaler(ScalerOptions{Centering: true})
  imputer.Fit(df)
  df = df.IndexView([]int{1, 2, 3, 4})
  err := imputer.TransformInplace(df)
  if err != nil {
    panic(err.Error())
  }
  vals := df.Floats("col")
  u.AssertFloatEquals("vals[0]", vals.Get(0), -0.5, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), -0.5, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 0.5, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 0.5, t)

  imputer.InverseTransformInplace(df)
  u.AssertFloatEquals("vals[0]", vals.Get(0), 1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 1.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 2.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 2.0, t)
}

func TestScalingAlone(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddFloats("col", 1.0, 1.0, 2.0, 2.0, math.NaN()).ToDataFrame()
  imputer := NewScaler(ScalerOptions{Scaling: true})
  imputer.Fit(df)
  df = df.IndexView([]int{0, 1, 2, 3})
  err := imputer.TransformInplace(df)
  if err != nil {
    panic(err.Error())
  }
  vals := df.Floats("col")
  u.AssertFloatEquals("vals[0]", vals.Get(0), 2.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 2.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 4.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 4.0, t)

  // reverse
  imputer.InverseTransformInplace(df)
  u.AssertFloatEquals("vals[0]", vals.Get(0), 1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 1.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 2.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 2.0, t)
}

func TestCenteringScaling(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddFloats("col", 1.0, 1.0, 2.0, 2.0, math.NaN()).ToDataFrame()
  imputer := NewScaler(ScalerOptions{Centering: true, Scaling: true})
  imputer.Fit(df)
  df = df.IndexView([]int{0, 1, 2, 3})
  err := imputer.TransformInplace(df)
  if err != nil {
    panic(err.Error())
  }
  vals := df.Floats("col")
  u.AssertFloatEquals("vals[0]", vals.Get(0), -1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), -1.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 1.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 1.0, t)

  // reverse
  imputer.InverseTransformInplace(df)
  u.AssertFloatEquals("vals[0]", vals.Get(0), 1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 1.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 2.0, t)
  u.AssertFloatEquals("vals[3]", vals.Get(3), 2.0, t)
}
