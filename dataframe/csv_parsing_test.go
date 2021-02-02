package dataframe

import (
  "os"
  "io/ioutil"
  "testing"
  u "github.com/rom1mouret/ml-essentials/utils"
)

func Test1FromCSVFile(t *testing.T) {
  f, err := ioutil.TempFile("", "test.*.csv")
	if err != nil {
    panic(err)
	}
  name := f.Name()
  defer os.Remove(name) // clean up

  // write a CSV with header
  f.WriteString("col1,col2,col3,col4\n")
  f.WriteString("0,0,1,0\n")       // 0    0  1    0
  f.WriteString("2.0,1,str,3\n")   // 2.0  1  str  3
  f.WriteString("3.0,1,str,3\n")   // 3.0  1  str  3
  f.WriteString("-,0,-,-\n")       // -    0  -    -
  f.Close()

  // test 1: with '-' as missing value
  data, err := FromCSVFile(name, CSVReadingSpec{MissingValues: []string{"-"}, MaxCPU: 2})
  if u.AssertTrue("error", err == nil, t) {
    if data.CheckConsistency(t) {
      u.AssertIntEquals("NumRows", data.NumAllocatedRows(), 4, t)
      u.AssertStringSliceEquals("cols", data.FloatHeader().NameList(), []string{"col1"}, false, t)
      u.AssertStringSliceEquals("cols", data.BoolHeader().NameList(), []string{"col2"}, false, t)
      u.AssertStringSliceEquals("cols", data.ObjectHeader().NameList(), []string{"col3"}, false, t)
      u.AssertStringSliceEquals("cols", data.StringHeader().NameList(), []string{"col3"}, false, t)
      u.AssertStringSliceEquals("cols", data.IntHeader().NameList(), []string{"col4"}, false, t)
    }
  }

  // test 2: without missing value
  data, err = FromCSVFile(name, CSVReadingSpec{MaxCPU: 2})
  if u.AssertTrue("error", err == nil, t) {
    if data.CheckConsistency(t) {
      expected := []string{"col1", "col3", "col4"}
      u.AssertIntEquals("NumRows", data.NumAllocatedRows(), 4, t)
      u.AssertStringSliceEquals("cols", data.ObjectHeader().NameList(), expected, false, t)
      u.AssertStringSliceEquals("cols", data.StringHeader().NameList(), expected, false, t)
    }
  }

  // test 3: without header
  opt := CSVReadingSpec{
      Header: []string{"c1", "c2", "c3", "c4"},
      MissingValues: []string{"-"},
      MaxCPU: 2}
  data, err = FromCSVFile(name, opt)
  if u.AssertTrue("error", err == nil, t) {
    expected := []string{"c1", "c2", "c3", "c4"}
    if data.CheckConsistency(t) {
      u.AssertIntEquals("NumRows", data.NumAllocatedRows(), 5, t)
      u.AssertStringSliceEquals("cols", data.ObjectHeader().NameList(), expected, false, t)
      u.AssertStringSliceEquals("cols", data.StringHeader().NameList(), expected, false, t)
    }
  }
}
