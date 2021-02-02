package preprocessing

import (
  "testing"
  "encoding/json"
  "github.com/rom1mouret/ml-essentials/dataframe"
  u "github.com/rom1mouret/ml-essentials/utils"
)

func rowSum(row []float32) float64 {
  sum := 0.0
  for _, v := range row {
    sum += float64(v)
  }
  return sum
}

func Test1HotBasics(t *testing.T) {
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddInts("col", 1, 2, 3, 2, 2, 2).ToDataFrame()
  encoder := NewOneHotEncoder(OneHotOptions{MissingPolicy: ReturnError, UnknownPolicy: ReturnError})
  encoder.Fit(df)
  u.AssertIntEquals("num new cols", len(encoder.NewColumns), 3, t)

  // serialization
  serialized, _ := json.Marshal(encoder)
  encoder = &OneHotEncoder{}
  json.Unmarshal([]byte(serialized), &encoder)

  df, err := encoder.TransformView(df)
  if err != nil {
    t.Errorf(err.Error())
    return
  }
  // number of columns
  nCols := len(df.BoolHeader().NameSet())
  if !u.AssertIntEquals("num output cols", nCols, 3, t) {
    return
  }
  // check the data with a row iterator
  twoIndex := -1
  ite := dataframe.NewFloat32Iterator(df, encoder.NewColumns)
  for row, i, _ := ite.NextRow(); row != nil;  row, i, _ = ite.NextRow() {
    u.AssertFloatEquals("row-sum", rowSum(row), 1, t)
    if i == 1 {
      for j, v := range row {
        if v > 0.9999 {
          twoIndex = j
          break
        }
      }
    } else if i >= 3 {
      for j, v := range row {
        if v > 0.9999 {
          u.AssertIntEquals("index", j, twoIndex, t)
        }
      }
    }
  }
}

func TestSeparateCategory(t *testing.T) {
  opt := OneHotOptions{MissingPolicy: SeparateCategory, UnknownPolicy: ReturnError}
  secondVal := -1
  for n := 1; n <= 2; n++ {
    builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
    df := builder.AddInts("col", 1).ToDataFrame()
    encoder := NewOneHotEncoder(opt)
    encoder.Fit(df)
    u.AssertIntEquals("num new cols", len(encoder.NewColumns), 2, t)

    df = builder.AddInts("col", 1, secondVal).ToDataFrame()
    df, err := encoder.TransformView(df)
    if err != nil {
      t.Errorf(err.Error())
      return
    }
    // number of columns
    nCols := len(df.BoolHeader().NameSet())
    if !u.AssertIntEquals("num output cols", nCols, 2, t) {
      return
    }
    // check the data with a row iterator
    ite := dataframe.NewFloat32Iterator(df, encoder.NewColumns)
    row1, _, _ := ite.NextRow()
    u.AssertFloatEquals("row2-sum", rowSum(row1), 1, t)
    row2, _, _ := ite.NextRow()
    u.AssertFloatEquals("row2-sum", rowSum(row2), 1, t)

    // other test
    opt = OneHotOptions{MissingPolicy: ReturnError, UnknownPolicy: SeparateCategory, KeepUsedColumns: false}
    secondVal = 10
  }
}

func Test1SeparateCategoryIfSeen(t *testing.T) {
  opt := OneHotOptions{MissingPolicy: SeparateCategoryIfSeen, UnknownPolicy: ReturnError}
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddInts("col", 1, 2).ToDataFrame()
  encoder := NewOneHotEncoder(opt)
  encoder.Fit(df)
  u.AssertIntEquals("num new cols", len(encoder.NewColumns), 2, t)
}

func Test2SeparateCategoryIfSeen(t *testing.T) {
  opt := OneHotOptions{MissingPolicy: SeparateCategoryIfSeen, UnknownPolicy: ReturnError}
  builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
  df := builder.AddInts("col", 1, 2, -1, -1).ToDataFrame()
  encoder := NewOneHotEncoder(opt)
  encoder.Fit(df)
  u.AssertIntEquals("num new cols", len(encoder.NewColumns), 3, t)
}

func TestMostFrequent(t *testing.T) {
  opt := OneHotOptions{MissingPolicy: ImputeWithMostFrequent, UnknownPolicy: ImputeWithMostFrequent}
  secondVal := -1
  for n := 1; n <= 2; n++ {
    builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
    df := builder.AddInts("col1", 1).AddInts("col2", 2).ToDataFrame()
    df.SetMaxCPU(3)
    encoder := NewOneHotEncoder(opt)
    encoder.Fit(df)
    u.AssertIntEquals("num new cols", len(encoder.NewColumns), 2, t)

    df = builder.AddInts("col1", 1, secondVal).AddInts("col2", 2, secondVal).ToDataFrame()
    df, err := encoder.TransformView(df)
    if err != nil {
      t.Errorf(err.Error())
      return
    }
    // number of columns
    nCols := len(df.BoolHeader().NameSet())
    u.AssertIntEquals("num output cols", nCols, 2, t)

    // other test
    secondVal = 10
    opt = OneHotOptions{MissingPolicy: ImputeWithMostFrequent, UnknownPolicy: SharedSeparateCategory, KeepUsedColumns: false}
  }
}

func TestErrorCatching(t *testing.T) {
  opt := OneHotOptions{MissingPolicy: ReturnError, UnknownPolicy: ReturnError}
  secondVal := -1
  for n := 1; n <= 2; n++ {
    builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
    df := builder.AddInts("col", 1).ToDataFrame()
    encoder := NewOneHotEncoder(opt)
    encoder.Fit(df)
    u.AssertIntEquals("num new cols", len(encoder.NewColumns), 1, t)

    df = builder.AddInts("col", 1, secondVal).ToDataFrame()
    _, err := encoder.TransformView(df)
    u.AssertTrue("error", err != nil, t)

    // other test
    secondVal = 10
  }
}
