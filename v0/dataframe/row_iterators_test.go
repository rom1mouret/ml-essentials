package dataframe

import (
    "testing"
    u "github.com/rom1mouret/ml-essentials/v0/utils"
)

func Test1Float32Iterator(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  ints := u.MakeRange(0, 1000, 1)
  builder.AddInts("ints1", ints...)
  builder.AddFloats("floats1", make([]float64, len(ints))...)
  builder.AddBools("bools1", make([]bool, len(ints))...)
  df := builder.ToDataFrame()

  iterator := NewFloat32Iterator(df, []string{"floats1", "bools1", "ints1"})
  i := 0
  for row, j, k := iterator.NextRow(); row != nil; row, j, k = iterator.NextRow() {
     u.AssertIntEquals("row number", i, j, t)
     u.AssertIntEquals("raw row number", j, k, t)
     u.AssertIntEquals("data", int(row[2]), i, t)
     i++
  }
  u.AssertIntEquals("num iterations", i, len(ints), t)
}
