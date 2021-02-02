package dataframe

import (
    "testing"
    u "github.com/rom1mouret/ml-essentials/utils"
)

func Test1ColumnNames(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("float1", 1.0, 2.0, 3.0)
  builder.AddFloats("float2", 4.0, 5.0, 6.0)
  builder.AddInts("int1", 1, 2, 3)
  builder.AddInts("int2", 4, 5, 6)
  builder.AddObjects("obj1", nil, "2", "3")
  builder.AddObjects("obj2", "4", "5", nil)
  builder.AddBools("bool1", false, true, false)
  builder.AddBools("bool2", true, false, true)
  builder.MarkAsString("obj1")
  expected := []string{"float1", "float2", "int1", "int2", "obj1", "obj2", "bool1", "bool2"}

  df := builder.ToDataFrame()
  df.PrintSummary().PrintHead(3, "")
  u.AssertStringSliceEquals("columns", df.Header().NameList(), expected, false, t)
}

func Test2ColumnNames(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("float1", 1.0, 2.0, 3.0)
  builder.AddFloats("float2", 4.0, 5.0, 6.0)
  builder.AddBools("bool1", false, true, false)
  builder.AddBools("bool2", true, false, true)
  expected := []string{"float1", "float2", "bool1", "bool2"}

  df := builder.ToDataFrame()
  u.AssertStringSliceEquals("columns", df.Header().NameList(), expected, false, t)
}

func TestShallowCopy(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("floats", 1.0, 2.0, 3.0, 4.0)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)
  df = df.IndexView([]int{0, 1})
  cpy := df.ShallowCopy()
  cpy.CheckConsistency(t)

  // everything is transferred properly
  u.AssertIntEquals("MaxCPU", cpy.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("NumRows", cpy.NumRows(), 2, t)
  u.AssertIntEquals("MaskSize", len(cpy.EmptyMask()), 2, t)

  // changing the shallow copy changes the original
  df.floats["floats"][df.indices[0]] = -1.0
  actual := cpy.floats["floats"][cpy.indices[0]]
  u.AssertFloatEquals("first value", actual,  -1.0, t)

  // adding columns to the shallow copy doesn't change the original
  cpy.floats["newcol"] = make([]float64, 1)  // kids, don't do this at home!
  colSet := df.Header().NameSet()
  u.AssertFalse("original df must be left alone", colSet["newcol"], t)
}

func TestCopy(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 1, 2, 3, 4)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)
  cpy := df.Copy()
  cpy.CheckConsistency(t)

  // everything is transferred properly
  u.AssertIntEquals("MaxCPU", cpy.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("NumRows", cpy.NumRows(), 4, t)
  u.AssertIntEquals("MaskSize", len(cpy.EmptyMask()), 4, t)
  u.AssertIntSliceEquals("data", df.ints["col"], []int{1, 2, 3, 4}, t)

  // changing the shallow copy doesn't changes the original
  df.ints["col"][df.indices[0]] = -1
  actual := cpy.ints["col"][cpy.indices[0]]
  u.AssertIntNotEquals("first value", actual,  -1, t)
}
