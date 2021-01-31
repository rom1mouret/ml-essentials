package dataframe

import (
   "github.com/rom1mouret/ml-essentials/v0/utils"
)

// DataFrame is a structure that lets you manipulate both original data and
// views on other dataframes' data by sharing the underlying data.
// The data is ordered by column.
type DataFrame struct {
  RawData
  indices      []int  // view on the data
  mask         []bool // just pre-allocated data, not a view on the data
  indexViewed  bool
  mThreadSafe  bool
  debug        bool // whether the debug mode is ON
  uid          int  // for debugging and PrintSummary
}

// EmptyDataFrame creates a new dataframe with no columns.
// maxCPU indicates how many CPUs are allowed to be utilized by the functions
// operating on the dataframe.
func EmptyDataFrame(nRows int, maxCPU int) *DataFrame {
  result := new(DataFrame)
  result.objects = make(map[string][]interface{})
  result.floats = make(map[string][]float64)
  result.bools = make(map[string][]bool)
  result.ints = make(map[string][]int)
  result.maxCPU = maxCPU
  result.dataUID = generateDataUID()
  result.resetStructureUID()
  // stringHeader is not pre-allocated

  if nRows > 0 {
    result.mask = make([]bool, nRows)
    result.indices = utils.MakeRange(0, nRows, 1)
  }
  return result
}

// ZeroMask returns a possibly pre-allocated mask for the MaskView function.
// The values of the mask are all initialized to false.
// Intended use:
// m := df.ZeroMask()
// for i := 0; i < df.NumRows(); i++ {
//    if i % 10 == 0 {
//       m[i] = true
//    }
// }
// df = df.MaskView(m)
// Do not concurrently use this function unless you call ThreadSafeMasking(True)
// first.
func (df *DataFrame) ZeroMask() []bool {
  nRows := df.NumRows()
  if df.mThreadSafe {
    mask := make([]bool, nRows)
    return mask
  }
  for i := 0; i < nRows; i++ {
    df.mask[i] = false
  }
  return df.mask[:nRows]
}

// EmptyMask returns a possibly pre-allocated mask for the MaskView function.
// The values of the mask are not initialized and can be either true of false.
// Intended use:
// m := df.EmptyMask()
// for i := 0; i < df.NumRows(); i++ {
//    m[i] = i % 10
// }
// df = df.MaskView(m)
// Do not concurrently use this function unless you call ThreadSafeMasking(True)
// first.
func (df *DataFrame) EmptyMask() []bool {
  if df.mThreadSafe {
    mask := make([]bool, df.NumRows())
    return mask
  }
  return df.mask[:df.NumRows()]
}

// NumRows returns the number of rows in the dataframe.
func (df *DataFrame) NumRows() int {
  return len(df.indices)
}

// ThreadSafeMasking makes the current dataframe and its views safe for masking.
// It returns the dataframe itself.
func (df *DataFrame) ThreadSafeMasking(enable bool) *DataFrame {
  df.mThreadSafe = enable
  return df
}

// Debug enables or disable the debugging mode.
// The debugging mode will print out some troubleshooting information via
// golang's builtin logger.
// It returns the dataframe itself.
func (df *DataFrame) Debug(enable bool) *DataFrame {
  df.debug = enable
  return df
}

// ShallowCopy copies the dataframe's structure but not the data.
// In 90% of cases, you would rather use View(), which doesn't even copy the
// structure up until the structure is modified.
// ShallowCopy returns a view on the dataframe.
func (df *DataFrame) ShallowCopy() *DataFrame {
  df.debugPrint("shallow-copying")
  result := EmptyDataFrame(-1, df.maxCPU)
  result.debug = df.debug
  result.mThreadSafe = df.mThreadSafe
  result.indices = df.indices
  result.mask = df.mask
  result.indexViewed = df.indexViewed
  result.TransferRawDataFrom(&df.RawData)
  result.debugPrint("ShallowCopy() returns")

  return result
}

// Copy returns a deep-copy of everything inside the dataframe, except the
// objects inside the object columns, despite copying the object slices.
// Call this function if you want to transform a view into a compact dataframe.
// Compact dataframes are more efficient, but making a copy can be expensive.
func (df *DataFrame) Copy() *DataFrame {
  df.debugPrint("copy")
  result := EmptyDataFrame(len(df.indices), df.maxCPU)
  result.mThreadSafe = df.mThreadSafe
  result.debug = df.debug
  result.stringHeader = df.stringHeader.Copy()

  if df.indexViewed {
    for col, v := range df.objects {
      series := make([]interface{}, len(df.indices))
      for j, index := range df.indices {
        series[j] = v[index]
      }
      result.objects[col] = series
    }
    for col, v := range df.floats {
      series := make([]float64, len(df.indices))
      for j, index := range df.indices {
        series[j] = v[index]
      }
      result.floats[col] = series
    }
    for col, v := range df.ints {
      series := make([]int, len(df.indices))
      for j, index := range df.indices {
        series[j] = v[index]
      }
      result.ints[col] = series
    }
    for col, v := range df.bools {
      series := make([]bool, len(df.indices))
      for j, index := range df.indices {
        series[j] = v[index]
      }
      result.bools[col] = series
    }
  } else {
    // that case is faster because indices = range(nRows)
    for col, v := range df.objects {
      series := make([]interface{}, len(v))
      copy(series, v)
      result.objects[col] = series
    }
    for col, v := range df.floats {
      series := make([]float64, len(v))
      copy(series, v)
      result.floats[col] = series
    }
    for col, v := range df.ints {
      series := make([]int, len(v))
      copy(series, v)
      result.ints[col] = series
    }
    for col, v := range df.bools {
      series := make([]bool, len(v))
      copy(series, v)
      result.bools[col] = series
    }
  }
  result.debugPrint("Copy() returns")
  return result
}
