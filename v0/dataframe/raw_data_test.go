package dataframe

import (
    "testing"
    u "github.com/rom1mouret/ml-essentials/v0/utils"
)


func TestUnshare(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3)
  df := builder.ToDataFrame()
  df2 := df.IndexView([]int{0, 2})
  df2.Unshare("col")
  df2.ints["col"][0] = 10
  u.AssertIntEquals("data[0]", df.ints["col"][0], 0, t)
}

func TestDrop(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddStrings("col", "a", "b", "c")
  df := builder.ToDataFrame()
  df.Drop("col")
  header := df.StringHeader().NameList()
  u.AssertIntEquals("num strings", len(header), 0, t)
}
