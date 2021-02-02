package preprocessing

import (
  "testing"
  "math"
  "github.com/rom1mouret/ml-essentials/dataframe"
  u "github.com/rom1mouret/ml-essentials/utils"
)

func TestMeanImputing(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddFloats("col", 1.0, 1.0, 1.0, math.NaN(), 3.0, 3.0).ToDataFrame()
  imputer := NewFloatImputer(FloatImputerOptions{Policy: Mean})
  df = df.IndexView([]int{0, 3, 4})
  imputer.Fit(df)
  err := imputer.TransformInplace(df)
  if err != nil {
    panic(err.Error())
  }
  vals := df.Floats("col")
  u.AssertFloatEquals("vals[0]", vals.Get(0), 1.0, t)
  u.AssertFloatEquals("vals[1]", vals.Get(1), 2.0, t)
  u.AssertFloatEquals("vals[2]", vals.Get(2), 3.0, t)
}
