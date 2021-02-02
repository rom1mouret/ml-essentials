package dataframe

import (
    "fmt"
)

type RowIterator struct {
  FloatBatching
  df        *DataFrame
  rowOffset int
  dfIndex   int
  subInds   []int
}

// Float32Iterator is a structure to iterate over a dataframe one row at a time.
// The rows provided to the user will be slices of float32.
// Float32Iterator cannot iterate through object columns.
// The delivered rows can be safely changed with no effect on the dataframe.
type Float32Iterator struct {
  RowIterator
  rows      [128][]float32
}

// Columns returns the name of the columns ordered like the row elements are
// ordered.
// If called on Float32Iterator or Float64Iterator, it returns the list of
// columns passed to NewFloat32Iterator and NewFloat64Iterator respectively.
func (ite *RowIterator) Columns() []string {
  return ite.columns
}

// Reset recycles the iterator's pre-allocated data for another dataframe with
// the same columns.
// If check is true, it will be verified that the columns are the same.
func (ite *RowIterator) Reset(df *DataFrame, check bool) {
  if check {
    for _, col := range ite.bColumns {
      colName := ite.columns[col]
      if _, ok := df.bools[colName]; !ok {
        panic(fmt.Sprintf("%s missing from bool columns", colName))
      }
    }
    for _, col := range ite.iColumns {
      colName := ite.columns[col]
      if _, ok := df.ints[colName]; !ok {
        panic(fmt.Sprintf("%s missing from int columns", colName))
      }
    }
    for _, col := range ite.fColumns {
      colName := ite.columns[col]
      if _, ok := df.floats[colName]; !ok {
        panic(fmt.Sprintf("%s missing from float columns", colName))
      }
    }
  }
  ite.rowOffset = 0
  ite.dfIndex = 0
  ite.df = df
}

func (ite *RowIterator) nextIndices() []int {
  end := ite.dfIndex + 128
  if end > len(ite.df.indices) {
    end = len(ite.df.indices)
  }
  ite.subInds = ite.df.indices[ite.dfIndex:end]

  return ite.subInds
}

// NewFloat32Iterator allocates a new row iterator to allow you to iterate
// over float, bool and int columns as floats.
// If a given column is not float, bool or int, it will be ignored.
// Row elements will be delivered in the same order as the columns passed as
// argument.
func NewFloat32Iterator(df *DataFrame, columns []string) *Float32Iterator {
  ite := new(Float32Iterator)
  ite.initialize(df, columns)
  ite.df = df
  // pre-allocation of the data
  for i, _ := range ite.rows {
    ite.rows[i] = make([]float32, len(ite.columns))
  }
  return ite
}

// NextRow returns a single row, its index in the view and its index in the
// original data. If there is no more row, it returns nil, the size of the view
// and the size of the original data.
// You can safely change the values of the row since they are copies of the
// original data. However, NextRow recycles the float slice, so you shouldn't
// store the slice.
func (ite *Float32Iterator) NextRow() ([]float32, int, int) {
  if ite.dfIndex == len(ite.df.indices) {
    return nil, len(ite.df.indices), ite.df.NumAllocatedRows()
  }
  if ite.rowOffset == 0 {
    indices := ite.RowIterator.nextIndices()
    // bools
    for _, colIx := range ite.bColumns {
      vals := ite.df.bools[ite.columns[colIx]]
      for j, i := range indices {
        if vals[i] {
          ite.rows[j][colIx] = 1.0
        } else {
          ite.rows[j][colIx] = 0
        }
      }
    }
    // ints
    for _, colIx := range ite.iColumns {
      vals := ite.df.ints[ite.columns[colIx]]
      for j, i := range indices {
        ite.rows[j][colIx] = float32(vals[i])
      }
    }
    // float64
    for _, colIx := range ite.fColumns {
      vals := ite.df.floats[ite.columns[colIx]]
      for j, i := range indices {
        ite.rows[j][colIx] = float32(vals[i])
      }
    }
  }
  row := ite.rows[ite.rowOffset]
  idx := ite.subInds[ite.rowOffset]
  ite.rowOffset = (ite.rowOffset + 1) % len(ite.rows)
  ite.dfIndex++

  return row, ite.dfIndex-1, idx
}
