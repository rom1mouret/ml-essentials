package dataframe

import (
    "testing"
    "strconv"
    "math/rand"
    u "github.com/rom1mouret/ml-essentials/utils"
)


func TestIndexView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  // select ints 9, 8, 7, 5, 6
  df = df.IndexView([]int{9, 8, 7, 6, 5})
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // select ints 9, 7, 5
  df = df.IndexView([]int{0, 2, 4})
  df.CheckConsistency(t)

  // check
  df.CheckConsistency(t)
  cpy := df.Copy()
  cpy.CheckConsistency(t)
  vals := cpy.ints["col"]
  u.AssertIntSliceEquals("int column", vals, []int{9, 7, 5}, t)
}

func TestIndexViewWithReps(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 41, 42)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  // select ints 42, 41, 42, 41, 42
  df = df.IndexView([]int{1, 0, 1, 0, 1})
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // check
  df.CheckConsistency(t)
  cpy := df.Copy()
  cpy.CheckConsistency(t)
  vals := cpy.ints["col"]
  u.AssertIntSliceEquals("int column", vals, []int{42, 41, 42, 41, 42}, t)
  u.AssertIntEquals("num rows", cpy.NumRows(), 5, t)
}

func TestSliceView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  // select ints 2, 3, 4, 5
  df = df.SliceView(2, 6)
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // select ints 4, 5
  df = df.SliceView(2, -1)
  df.CheckConsistency(t)

  // check
  df.CheckConsistency(t)
  cpy := df.Copy()
  cpy.CheckConsistency(t)
  vals := cpy.ints["col"]
  u.AssertIntSliceEquals("int column", vals, []int{4, 5}, t)
}

func Test1MaskView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  // selects 0, 2, 4
  mask := df.ZeroMask()
  mask[0] = true
  mask[2] = true
  mask[4] = true
  df = df.MaskView(mask)
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // select ints 0, 4
  mask = df.EmptyMask()
  mask[0] = true
  mask[1] = false
  mask[2] = true
  df = df.MaskView(mask)
  df.CheckConsistency(t)

  // check
  df.CheckConsistency(t)
  cpy := df.Copy()
  cpy.CheckConsistency(t)
  vals := cpy.ints["col"]
  u.AssertIntSliceEquals("int column", vals, []int{0, 4}, t)
}

func Test2MaskView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  // new ints = 1, 1, 1, 0
  df = df.IndexView([]int{1, 1, 1, 0})

  // select ints 1, 0
  mask := df.ZeroMask()
  mask[0] = true
  mask[len(mask)-1] = true
  df = df.MaskView(mask)
  df.CheckConsistency(t)

  // check
  df.CheckConsistency(t)
  cpy := df.Copy()
  cpy.CheckConsistency(t)
  vals := cpy.ints["col"]
  u.AssertIntSliceEquals("int column", vals, []int{1, 0}, t)
}

func TestColumnView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col1", 1, 2, 3)
  builder.AddStrings("col2", "one", "two", "three")
  df := fillBlanks(builder)
  df.SetMaxCPU(1)
  df = df.ColumnView("col2")

  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  df.CheckConsistency(t)
  u.AssertStringSliceEquals("col2 exists", df.StringHeader().NameList(), []string{"col2"}, true, t)
  u.AssertIntEquals("no col1", len(df.ints), 0, t)
}

func TestShuffleView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  df = df.ShuffleView()
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  df.CheckConsistency(t)
  u.AssertIntSliceEquals("data hasn't changed", df.ints["col"], []int{0, 1, 2, 3, 4, 5}, t)

  cpy := df.Copy()
  u.AssertIntSliceNotEquals("not same order", cpy.ints["col"], []int{0, 1, 2, 3, 4, 5}, t)
}

func TestSampleView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  df = df.SampleView(3, false)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  df.CheckConsistency(t)
  u.AssertIntSliceEquals("data hasn't changed", df.ints["col"], []int{0, 1, 2, 3, 4, 5}, t)

  cpy := df.Copy()
  u.AssertIntEquals("num rows", len(cpy.ints["col"]), 3, t)
}

func Test1SplitNView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  dfs := df.SplitNView(3)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("num dataframes", len(dfs), 3, t)
  for _, view := range dfs {
    view.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", view.ActualMaxCPU(), 1, t)
  }
  u.AssertIntEquals("NumRows(dfs[0])", dfs[0].NumRows(), 2, t)
  u.AssertIntEquals("NumRows(dfs[1])", dfs[1].NumRows(), 2, t)
  u.AssertIntEquals("NumRows(dfs[2])", dfs[2].NumRows(), 1, t)

  cpy1 := dfs[0].Copy()
  cpy2 := dfs[1].Copy()
  cpy3 := dfs[2].Copy()
  data := append(append(cpy1.ints["col"], cpy2.ints["col"]...), cpy3.ints["col"]...)

  u.AssertIntSliceEquals("data", data, []int{0, 1, 2, 3, 4}, t)
}

func Test2SplitNView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4, 5)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  dfs := df.SplitNView(3)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("num dataframes", len(dfs), 3, t)
  for _, view := range dfs {
    view.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", view.ActualMaxCPU(), 1, t)
  }
  u.AssertIntEquals("NumRows(dfs[0])", dfs[0].NumRows(), 2, t)
  u.AssertIntEquals("NumRows(dfs[1])", dfs[1].NumRows(), 2, t)
  u.AssertIntEquals("NumRows(dfs[2])", dfs[2].NumRows(), 2, t)

  cpy1 := dfs[0].Copy()
  cpy2 := dfs[1].Copy()
  cpy3 := dfs[2].Copy()
  data := append(cpy1.ints["col"], cpy2.ints["col"]...)
  data = append(data, cpy3.ints["col"]...)

  u.AssertIntSliceEquals("data", data, []int{0, 1, 2, 3, 4, 5}, t)
}

func Test3SplitNView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  dfs := df.SplitNView(3)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("num dataframes", len(dfs), 3, t)
  for _, view := range dfs {
    view.CheckConsistency(t)
    u.AssertIntEquals("MaxCPU", view.ActualMaxCPU(), 1, t)
  }
  u.AssertIntEquals("NumRows(dfs[0])", dfs[0].NumRows(), 1, t)
  u.AssertIntEquals("NumRows(dfs[1])", dfs[1].NumRows(), 1, t)
  u.AssertIntEquals("NumRows(dfs[2])", dfs[2].NumRows(), 0, t)

  cpy1 := dfs[0].Copy()
  cpy2 := dfs[1].Copy()
  cpy3 := dfs[2].Copy()
  data := append(cpy1.ints["col"], cpy2.ints["col"]...)
  data = append(data, cpy3.ints["col"]...)

  u.AssertIntSliceEquals("data", data, []int{0, 1}, t)
}

func TestSplitTrainTestViews(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3, 4)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  df1, df2 := df.SplitTrainTestViews(0.5)
  df1.CheckConsistency(t)
  df2.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df1.ActualMaxCPU(), 1, t)
  u.AssertIntEquals("MaxCPU", df2.ActualMaxCPU(), 1, t)

  cpy1 := df1.Copy()
  cpy2 := df2.Copy()
  data := append(cpy1.ints["col"], cpy2.ints["col"]...)
  u.AssertIntSliceEquals("data", data, []int{0, 1, 2, 3, 4}, t)
}

func Test1SortedView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("col", 4.0, 1.0, 3.0, 2.0)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  df = df.SortedView("col")
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // data is sorted
  cpy := df.Copy()
  u.AssertFloatSliceEquals("data", cpy.floats["col"], []float64{1, 2, 3, 4}, t)

  // original data is kept intact
  u.AssertFloatSliceEquals("data", df.floats["col"], []float64{4, 1, 3, 2}, t)
}

func Test2SortedView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 4, 1, 3, 2)
  df := fillBlanks(builder)
  df = df.SortedView("col")
  df.CheckConsistency(t)

  // data is sorted
  cpy := df.Copy()
  u.AssertIntSliceEquals("data", cpy.ints["col"], []int{1, 2, 3, 4}, t)

  // original data is kept intact
  u.AssertIntSliceEquals("data", df.ints["col"], []int{4, 1, 3, 2}, t)
}

func Test3SortedView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddBools("col", false, true, true, false, false)
  df := fillBlanks(builder)
  df = df.SortedView("col")
  df.CheckConsistency(t)

  // data is sorted
  cpy := df.Copy()
  u.AssertBoolSliceEquals("data", cpy.bools["col"], []bool{false, false, false, true, true}, t)

  // original data is kept intact
  u.AssertBoolSliceEquals("data", df.bools["col"], []bool{false, true, true, false, false}, t)
}

func TestTopView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddFloats("col", 4.0, 1.0, 3.0, 2.0)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)

  df = df.TopView("col", 2, true, true)
  df.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)

  // data is sorted
  cpy := df.Copy()
  u.AssertFloatSliceEquals("data", cpy.floats["col"], []float64{1, 2}, t)

  // original data is kept intact
  u.AssertFloatSliceEquals("data", df.floats["col"], []float64{4, 1, 3, 2}, t)
}

func TestReverseView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 0, 1, 2, 3)
  df := fillBlanks(builder)
  df.SetMaxCPU(1)
  df = df.ReverseView()
  u.AssertIntEquals("MaxCPU", df.ActualMaxCPU(), 1, t)
  cpy := df.Copy()
  u.AssertIntSliceEquals("data", cpy.ints["col"], []int{3, 2, 1, 0}, t)
}

func TestHashStringsView(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddObjects("col", "one", "two", "three", nil)
  builder.MarkAsString("col")
  df := fillBlanks(builder)

  // reference
  df.SetMaxCPU(1)
  view := df.HashStringsView("col")
  view.PrintSummary()
  view.CheckConsistency(t)
  u.AssertIntEquals("MaxCPU", view.ActualMaxCPU(), 1, t)
  hashed := view.ints["col"]
  u.AssertIntEquals("hash(nil)", hashed[3], -1, t)
  for i, v := range hashed[1:] {
    u.AssertIntNotEquals("v[i]!=v[i-1]", v, hashed[i], t)
  }
  // try with different max CPU values
  for cpu := range u.MakeRange(-1, 6, 1) {
    df.SetMaxCPU(cpu)
    hashed2 := df.HashStringsView("col").ints["col"]
    u.AssertIntSliceEquals("hashed", hashed2, hashed, t)
  }
}

func TestNegativeHashStringsView(t *testing.T) {
  arr := make([]string, 30000)
  for i := 0; i < len(arr); i++ {
    arr[i] = strconv.Itoa(i)
  }
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddStrings("col", arr...)
  df := fillBlanks(builder)
  // reference
  view := df.HashStringsView("col")
  view.CheckConsistency(t)
  hashed := view.ints["col"]
  for _, v := range hashed {
    u.AssertTrue("v positive", v >= 0, t)
  }
}

func TestMess1(t *testing.T) {
  // large df to avoid collisions
  arr := make([]int, 1000000)
  for i := 0; i < len(arr); i++ {
    arr[i] = i % (len(arr) / 3)
  }
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("i", arr...)
  df := fillBlanks(builder)

  // some random stuff
  rand.Seed(0)
  nTests := 100
  for tests := 0; tests < nTests; tests++ {
    view := df
    for n := 0; n < 6; n++ {
      topN := 1 + 10000 * (6 - n)
      switch randVal := rand.Intn(8); randVal {
        case 0:
      		view = view.ShuffleView()
      	case 1:
      		view = view.ReverseView()
        case 2:
          view = view.SortedView("i")
        case 3:
          view = view.TopView("i", topN, false, false)
        case 4:
          view = view.TopView("i", topN, true, false)
        case 5:
          view = view.TopView("i", topN, false, true)
        case 6:
          view = view.TopView("i", topN, true, true)
      	default:
          view = view.SliceView(3, -1)
	    }
    }
    view.ShuffleView().Ints("i").Set(0, -1)
  }
  // count the number of -1
  actual := 0
  access := df.Ints("i")
  for i := 0; i < access.Size(); i++ {
    if access.Get(i) == -1 {
      actual++
    }
  }
  u.AssertIntEquals("count(-1)", actual, nTests, t)
}
