package dataframe

import (
  "os"
  "io/ioutil"
  "testing"
  u "github.com/rom1mouret/ml-essentials/utils"
)

func TestToMultiCSVFiles(t *testing.T) {
  // create two temporary files
  f, err := ioutil.TempFile("", "test.*.csv")
	if err != nil {
    panic(err)
	}
  name1 := f.Name()
  f.Close()
  defer os.Remove(name1) // clean up

  f, err = ioutil.TempFile("", "test.*.csv")
	if err != nil {
    panic(err)
	}
  name2 := f.Name()
  f.Close()
  defer os.Remove(name2) // clean up

  // testing
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddInts("col", 7, 8, 9)
  data := fillBlanks(builder)
  err = data.ToCSVFiles(CSVWritingSpec{}, name1, name2)
  data1, _ := FromCSVFile(name1, CSVReadingSpec{MaxCPU: 8})
  data2, _ := FromCSVFile(name2, CSVReadingSpec{MaxCPU: 8})

  if data1.CheckConsistency(t) && data2.CheckConsistency(t) {
    df1 := data1.ToDataFrame()
    df2 := data2.ToDataFrame()
    df, err := RowConcat(df1, df2)
    if err != nil {
      panic(err)
    }
    if df.CheckConsistency(t) {
      u.AssertIntSliceEquals("data", data.ints["col"], []int{7, 8, 9}, t)
    }
  }
}
