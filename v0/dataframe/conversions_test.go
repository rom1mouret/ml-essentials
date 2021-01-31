package dataframe

import (
    "testing"
    u "github.com/rom1mouret/ml-essentials/v0/utils"
)

func TestCopyValuesToInterfaces(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 1, 2, 3, 4)
  df := fillBlanks(builder).IndexView([]int{0, 1})

  vals := df.CopyValuesToInterfaces("col")
  df.CopyValuesToInterfaces("SomeFloats")
  df.CopyValuesToInterfaces("SomeBools")
  df.CopyValuesToInterfaces("SomeObjects")
  u.AssertIntEquals("len(vals)", len(vals), 2, t)
}

func TestLabelToInts(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddObjects("col", "one", "two", "three", nil, "two")
  builder.MarkAsString("col")
  df := fillBlanks(builder).IndexView([]int{0, 1, 3, 4})
  vals, mapping := df.LabelToInt("col")
  u.AssertIntEquals("num unique values", len(mapping), 2, t)
  u.AssertIntEquals("nil string mapping", vals[2], -1, t)
}
