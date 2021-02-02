package dataframe

import (
   "fmt"
   "github.com/rom1mouret/ml-essentials/utils"
)

// RowConcat concatenates the rows of the given dataframes.
// All the data is copied, i.e. the returned dataframe does not share any data
// or structure with the dataframes given as arguments.
// It returns an error if the dataframes don't have the same columns.
// Numpy equivalent: concat(dfs, axis=0)
func RowConcat(dfs ...*DataFrame) (*DataFrame, error) {
  if len(dfs) == 0 {
    return nil, fmt.Errorf("RowConcat requires at least one dataframe")
  }
  // total number of rows
  nRows := 0
  for _, df := range dfs {
    nRows += df.NumRows()
  }
  // allocation
  result := EmptyDataFrame(nRows, dfs[0].maxCPU)
  for col := range dfs[0].floats {
    result.floats[col] = make([]float64, nRows)
  }
  for col := range dfs[0].bools {
    result.bools[col] = make([]bool, nRows)
  }
  for col := range dfs[0].objects {
    result.objects[col] = make([]interface{}, nRows)
  }
  for col := range dfs[0].ints {
    result.ints[col] = make([]int, nRows)
  }
  // merge
  offset := 0
  for k, df := range dfs {
    result.stringHeader = result.stringHeader.And(df.stringHeader)
    // TODO: optimize this when !df.indexViewed
    for col, vals := range result.floats {
      if from, ok := df.floats[col]; ok {
        for j, i := range df.indices {
          vals[j + offset] = from[i]
        }
      } else {
        return nil, fmt.Errorf("float column %s not found in %dth dataframe", col, k)
      }
    }
    for col, vals := range result.ints {
      if from, ok := df.ints[col]; ok {
        for j, i := range df.indices {
          vals[j + offset] = from[i]
        }
      } else {
        return nil, fmt.Errorf("int column %s not found in %dth dataframe", col, k)
      }
    }
    for col, vals := range result.objects {
      if from, ok := df.objects[col]; ok {
        for j, i := range df.indices {
          vals[j + offset] = from[i]
        }
      } else {
        return nil, fmt.Errorf("object column %s not found in %dth dataframe", col, k)
      }
    }
    for col, vals := range result.bools {
      if from, ok := df.bools[col]; ok {
        for j, i := range df.indices {
          vals[j + offset] = from[i]
        }
      } else {
        return nil, fmt.Errorf("bool column %s not found in %dth dataframe", col, k)
      }
    }
    offset += df.NumRows()
  }
  return result, nil
}

// ColumnSmartConcat merges the columns from multiple dataframes.
// For example, if columns(df1)=[col1, col2] and columns(df2)=[col3] then
// columns(ColumnSmartConcat(df1, df2)) = [col1, col2, col3]
// It returns an error if the number of rows don't match.
// The returned dataframe shares data with the dataframes given as arguments,
// unless said dataframes' inner indices are not congruent.
// Use this function if you are not going to change the returned dataframe and
// want to avoid unnecessary copies when possible.
// Numpy equivalent: concat(dfs, axis=1)
func ColumnSmartConcat(dfs ...*DataFrame) (*DataFrame, error) {
  // major difference with pandas: it doesn't concatenate index-wise
  newDF, err := ColumnConcatView(dfs...)
  if err == nil {
    return newDF, nil
  }
  // TODO: if some dataframes are small, we can reset their indices

  return ColumnCopyConcat(dfs...)
}

// ColumnConcatView merges the columns from multiple dataframes.
// For example, if columns(df1)=[col1, col2] and columns(df2)=[col3] then
// columns(ColumnConcatView(df1, df2)) = [col1, col2, col3]
// It returns an error if the number of rows or the inner indices don't match.
// The returned dataframe shares data with the dataframes given as arguments,
// so changing the input dataframes will also change the returned dataframe.
// Numpy equivalent: concat(dfs, axis=1)
func ColumnConcatView(dfs ...*DataFrame) (*DataFrame, error) {
  err := CheckNoColumnOverlap(dfs)
  if err != nil {
    return nil, err
  }
  // check that all the indices match
  views := false
  for _, df := range dfs {
    if df.indexViewed {
      views = true
      break
    }
  }
  if views {
    ref := utils.HashIntSlice(dfs[0].indices)
    for _, df := range dfs[1:] {
      if utils.HashIntSlice(df.indices) != ref {
        return nil, fmt.Errorf("index mistmatch between dataframes. Use ColumnConcat instead")
      }
    }
  }
  // merge in a shallow way
  result := dfs[0].ShallowCopy()
  for _, df := range dfs[1:] {
    for col, values := range df.floats {
      result.floats[col] = values
      result.shared.add(col)
    }
    for col, values := range df.ints {
      result.ints[col] = values
      result.shared.add(col)
    }
    for col, values := range df.objects {
      result.objects[col] = values
      result.shared.add(col)
    }
    for col, values := range df.bools {
      result.bools[col] = values
      result.shared.add(col)
    }
    result.stringHeader.And(df.stringHeader)
  }
  return result, nil
}

// ColumnCopyConcat merges the columns from multiple dataframes.
// For example, if columns(df1)=[col1, col2] and columns(df2)=[col3] then
// columns(ColumnCopyConcat(df1, df2)) = [col1, col2, col3]
// It returns an error if the number of rows don't match.
// The returned dataframe doesn't share any data with the input dataframe,
// so the returned dataframe is safe to change.
// Numpy equivalent: concat(dfs, axis=1)
func ColumnCopyConcat(dfs ...*DataFrame) (*DataFrame, error) {
  // check the number of rows match
  ref := dfs[0].NumRows()
  for _, df := range dfs[1:] {
    n2 := df.NumRows()
    if n2 != ref {
      return nil, fmt.Errorf("cannot col-wise concatenate df(rows=%d) with df(rows=%d)", ref, n2)
    }
  }
  // make a copy of each dataframe
  copies := make([]*DataFrame, len(dfs))
  for i, df := range dfs {
    copies[i] = df.Copy()
  }
  // now we can safely call the view method
  result, err := ColumnConcatView(copies...)
  if err != nil {
    return nil, err
  }
  result.shared = ColumnHeader{}

  return result, nil
}
