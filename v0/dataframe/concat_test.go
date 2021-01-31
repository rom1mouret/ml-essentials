package dataframe

import (
    "testing"
    u "github.com/rom1mouret/ml-essentials/v0/utils"
)

func TestRowConcat(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 1, 2, 3)
  df1 := fillBlanks(builder)
  df1.SetMaxCPU(1)

  builder = DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 4, 5, 6)
  df2 := fillBlanks(builder)
  df2.SetMaxCPU(1)

  df, err := RowConcat(df1, df2)
  if u.AssertTrue("error", err == nil, t) {
    df.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
    data := df.ints["col"]
    u.AssertIntSliceEquals("data", data, []int{1, 2, 3, 4, 5, 6}, t)
  }
}

func TestColumnCopyConcat(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col1", 1, 2, 3)
  df1 := builder.ToDataFrame()
  builder = DataBuilder{RawData: NewRawData()}
  builder.AddInts("col2", 4, 5, 6)
  df2 := builder.ToDataFrame()
  df1.SetMaxCPU(1)
  df2.SetMaxCPU(1)

  df, err := ColumnCopyConcat(df1, df2)
  if u.AssertTrue("error", err == nil, t) {
    df.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
    data1 := df.ints["col1"]
    u.AssertIntSliceEquals("data", data1, []int{1, 2, 3}, t)
    data2 := df.ints["col2"]
    u.AssertIntSliceEquals("data", data2, []int{4, 5, 6}, t)
  }
}

func Test1ColumnConcatView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col1", 1, 2, 3)
  df1 := builder.ToDataFrame()
  builder = DataBuilder{RawData: NewRawData()}
  builder.AddInts("col2", 4, 5, 6)
  df2 := builder.ToDataFrame()
  df1.SetMaxCPU(1)
  df2.SetMaxCPU(1)

  df1 = df1.SliceView(-1, -2)
  df2 = df2.SliceView(-1, -2)
  df, err := ColumnConcatView(df1, df2)
  if u.AssertTrue("error", err == nil, t) {
    df.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  }
}

func Test2ColumnConcatView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col1", 1, 2, 3)
  df1 := builder.ToDataFrame()
  builder = DataBuilder{RawData: NewRawData()}
  builder.AddInts("col2", 4, 5, 6)
  df2 := builder.ToDataFrame()
  df1 = df1.IndexView([]int{0, 0, 0, 0})
  df2 = df2.IndexView([]int{1, 1, 1, 1})

  _, err := ColumnConcatView(df1, df2)
  u.AssertTrue("error", err != nil, t)
}
