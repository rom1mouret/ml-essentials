package dataframe

import (
  "fmt"
  "testing"
  "github.com/rom1mouret/ml-essentials/v0/utils"
)

// CheckNoColumnOverlap returns an error if two or more dataframes have one or
// more columns in common, regardless of their type.
// It returns nil if there is no overlap.
func CheckNoColumnOverlap(dfs []*DataFrame) error {
  colSet := make(map[string]bool)
  for k, df := range dfs {
    for col := range df.floats {
      if _, ok := colSet[col]; ok {
        return fmt.Errorf("column %s (%dth dataframe) overlaps", col, k)
      }
      colSet[col] = true
    }
    for col := range df.ints {
      if _, ok := colSet[col]; ok {
        return fmt.Errorf("column %s (%dth dataframe) overlaps", col, k)
      }
      colSet[col] = true
    }
    for col := range df.objects {
      if _, ok := colSet[col]; ok {
        return fmt.Errorf("column %s (%dth dataframe) overlaps", col, k)
      }
      colSet[col] = true
    }
    for col := range df.bools {
      if _, ok := colSet[col]; ok {
        return fmt.Errorf("column %s (%dth dataframe) overlaps", col, k)
      }
      colSet[col] = true
    }
  }
  return nil
}

func (data *RawData) CheckConsistency(t *testing.T) bool {
  nRows := data.NumAllocatedRows()
  for _, vals := range data.objects {
    if !utils.AssertIntEquals("object-column", len(vals), nRows, t) {
      return false
    }
  }
  for _, vals := range data.floats {
    if !utils.AssertIntEquals("float-column", len(vals), nRows, t) {
      return false
    }
  }
  for _, vals := range data.ints {
    if !utils.AssertIntEquals("int-column", len(vals), nRows, t) {
      return false
    }
  }
  for _, vals := range data.bools {
    if !utils.AssertIntEquals("bool-column", len(vals), nRows, t) {
      return false
    }
  }
  for _, col := range data.stringHeader.NameList() {
    if _, ok := data.objects[col]; !ok {
      utils.AssertTrue("string columns should be in objects", false, t)
      return false
    }
  }
  for col, vals := range data.objects {
    // check that all the non-string columns are not string
    // (it is not a requirement but our tests always run with this setup)
    if !data.stringHeader.NameSet()[col] {
      for _, v := range vals {
        if v != nil {
          s, ok := v.(string)
          if ok {
            msg := fmt.Sprintf("'%s' is a string in non-string %s col", s, col)
            if !utils.AssertFalse(msg, ok, t) {
              return false
            }
          }
        }
      }
    }
  }
  return true
}
