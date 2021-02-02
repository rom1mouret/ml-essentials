package dataframe

import (
    "gonum.org/v1/gonum/mat"
)

type Dense64Batching struct {
  FloatBatching
  data     []float64
}

// NewDense64Batching allocates a new Dense64Batching structure.
// Dense64Batching will copy the columns passed as arguments in the same order
// as given to this function.
// Dense64Batching recycles the data between successive calls to DenseMatrix, so
// try to call NewDense64Batching only once and DenseMatrix as many times as
// needed.
func NewDense64Batching(columns []string) *Dense64Batching {
  result := new(Dense64Batching)
  result.columns = columns
  return result
}

// DenseMatrix generates a gonum dense matrix from the given dataframe.
// The returned data is a copy of the dataframe's data, so changing the matrix
// doesn't change the dataframe.
// The returned matrix is stored as a transpose. Any operation on this matrix
// will be faster if it involves another transpose.
// Dense64Batching recycles the data between successive calls to DenseMatrix, so
// try to call NewDense64Batching only once and DenseMatrix as many times as
// needed.
func (bat *Dense64Batching) DenseMatrix(df *DataFrame) mat.Matrix {
  df.debugPrint("Dense64Batching.DenseMatrix")
  bat.initialize(df, bat.columns) // yes, bat.columns will be initialized again
  // allocate more data if necessary
  dim := len(bat.columns)
  nRows := df.NumRows()
  missing := dim * (nRows - len(bat.data))
  if missing > 0 {
    bat.data = append(bat.data, make([]float64, missing)...)
  }
  // copy the data in column-major order
  for _, colIx := range bat.fColumns {
    subslice := bat.data[nRows * colIx:]
    rawdata := df.floats[bat.columns[colIx]]
    if df.indexViewed {
      for i, j := range df.indices {
        subslice[i] = rawdata[j]
      }
    } else {
      copy(subslice, rawdata)
    }
  }
  for _, colIx := range bat.iColumns {
    subslice := bat.data[nRows * colIx:]
    rawdata := df.ints[bat.columns[colIx]]
    for i, j := range df.indices {
      subslice[i] = float64(rawdata[j])
    }
  }
  for _, colIx := range bat.bColumns {
    subslice := bat.data[nRows * colIx:]
    rawdata := df.bools[bat.columns[colIx]]
    for i, j := range df.indices {
      if rawdata[j] {
        subslice[i] = 1.0
      } else {
        subslice[i] = 0
      }
    }
  }
  // Pack everything
  // Note that we purposefully swap nRows and nCols prior to calling T()
  return mat.NewDense(dim, nRows, bat.data[:nRows * dim]).T()
}
