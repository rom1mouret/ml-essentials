package dataframe

import (
    "testing"
)

func TestDataBuilderExample(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("height", 170, 180, 165)
  builder.AddStrings("name", "Karen", "John", "Sophie")
  df := builder.ToDataFrame()
  df.PrintSummary().PrintHead(-1, "%.3f").PrintRecords(-1, "", df.GoodShortNames(0))
}

func TestColAccessExample(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("height", 170, 180, 165)
  builder.AddStrings("name", "Karen", "John", "Sophie")
  df := builder.ToDataFrame()

  height := df.Floats("height")
  for i := 0; i < height.Size(); i++ {
    height.Set(i, height.Get(i) / 2)
  }
}
