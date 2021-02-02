package dataframe

import (
  "fmt"
  "sort"
  "math/rand"
  "hash/fnv"
  "github.com/rom1mouret/ml-essentials/utils"
)

// AreIndicesAltered returns true if the internal list of indices is not
// range(0, df.NumRows())
func (df *DataFrame) AreIndicesAltered() bool {
  return df.indexViewed
}

// IndexView builds a view from a selection of rows.
// The given slice of indices is typically a subset of range(0, df.NumRows()),
// but it can also be a different order of range(0, df.NumRows()) or a
// repetition of some indices, thereby making the view larger than its parent
// dataframe.
// It is equivalent to x[indices] where x is a Python numpy array, except that
// IndexView doesn't do any copy.
func (df *DataFrame) IndexView(indices []int) *DataFrame {
  result := df.View()
  result.indices = make([]int, len(indices))
  for j, index := range indices {
    result.indices[j] = df.indices[index]
  }
  if len(df.indices) < len(indices) {
    // we extend the mask slice because it's smaller than NumRows
    // (a rare case)
    diff := len(indices) - len(df.indices)
    result.mask = append(result.mask, make([]bool, diff)...)
  }
  result.indexViewed = true

  return result
}

// SliceView builds a view from a slice of the dataframe from index "from"
// (included) to index "to" (excluded).
// If "from" or "to" is negative, the index is relative to the end of the
// dataframe. For example, -1 points to the last index of the dataframe.
// If "from" is higher than "to", the row order will be reversed.
func (df *DataFrame) SliceView(from int, to int) *DataFrame {
  if from < 0 {
    from += len(df.indices) + 1
  }
  if to < 0 {
    to += len(df.indices) + 1
  }
  if from > to {
    to, from = from, to
    df = df.ReverseView()
  }
  result := df.View().cut(from, to)

  return result
}

// MaskView builds a view by masking some rows of the dataframe.
// To avoid unnecessary allocations, please get a pre-allocated mask from
// DataFrame.EmptyMask() or DataFrame.ZeroMask().
// MaskView is functionally equivalent to:
//  indices = make([]int, 0)
//  for i, b := mask {
//    if b {
//      indices = append(indices, i)
//    }
//  }
//  maskedView := df.IndexView(indices)
func (df *DataFrame) MaskView(mask []bool) *DataFrame {
  result := df.View()
  result.indices = make([]int, len(mask))
  newSize := 0
  for j, b := range mask {
    if b {
      result.indices[newSize] = df.indices[j]
      newSize++
    }
  }
  result.indices = result.indices[:newSize]  // truncation
  result.indexViewed = true

  return result
}

// ColumnView selects a subset of columns.
func (df *DataFrame) ColumnView(columns ...string) *DataFrame {
  if len(columns) == 0 {
    return df.View()
  }
  result := df.View()
  result.allocateEmptyMaps()
  result.dataUID = df.dataUID

  // set of column to select
  colSet := utils.ToStringSet(columns)
  result.shared.add(columns...)

  // select the given subset
  stringCols := df.stringHeader.NameSet()
  for col, v := range df.objects {
    if colSet[col] {
      result.objects[col] = v
      if stringCols[col] {
        result.stringHeader.add(col)
      }
    }
  }
  for col, v := range df.floats {
    if colSet[col] {
      result.floats[col] = v
    }
  }
  for col, v := range df.bools {
    if colSet[col] {
      result.bools[col] = v
    }
  }
  for col, v := range df.ints {
    if colSet[col] {
      result.ints[col] = v
    }
  }
  return result
}

// ShuffleView randomizes the dataframe.
// This is functionally equivalent to this pseudo-code:
//  indices = range(0, df.NumRows())
//  shuffle(indices)
//  shuffledView = df.IndexView(indices)
// If you want ShuffleView to behave deterministically, you need to call
// rand.Seed(seed) somewhere in your program prior to calling ShuffleView.
func (df *DataFrame) ShuffleView() *DataFrame {
  a := utils.MakeRange(0, df.NumRows(), 1)
  rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
  return df.IndexView(a)
}

// SampleView randomly samples n rows from the dataframe.
// Sampling with replacement is not yet supported.
// Sampling without replacement is functionally equivalent to:
//  df.ShuffleView().SliceView(0, n)
func (df *DataFrame) SampleView(n int, replacement bool) *DataFrame {
  if replacement {
    panic("replacement is not supported yet")
  }
  if n > df.NumRows() {
    panic(fmt.Sprintf("sampling %d > size(df) = %d is not yet supported",
                      n, df.NumRows()))
  }
  // TODO: better algorithm
  df = df.ShuffleView()
  df.indices = df.indices[:n]

  return df
}

// SplitNView evenly divides the dataframe into n parts.
// It will panic if n is negative and returns nil if n equals zero.
// It always returns *exactly* n dataframes. As a result, some dataframes might
// be empty.
func (df *DataFrame) SplitNView(n int) []*DataFrame {
  if n < 0 {
    panic("DataFrames cannot be split into n<0 parts")
  }
  if n == 0 {
    return nil
  }
  if n == 1 || df.NumRows() == 0 {
    return []*DataFrame{df.View()}
  }
  // batchSize for the first n-1 batches
  nRows := df.NumRows()
  batchSize := nRows / n
  if nRows % n != 0 {
    batchSize++
  }
  result := df.SplitView(batchSize)
  missing := n - len(result)
  if missing > 0 {  // expected to happen when batchSize = 1
    // fill up the rest with empty DataFrames
    emptyDF := EmptyDataFrame(0, df.maxCPU)
    emptyDFs := make([]*DataFrame, missing)
    for i := 0; i < missing; i++ {
      emptyDFs[i] = emptyDF
    }
    result = append(result, emptyDFs...)
  }
  return result
}

// SplitView divides the dataframe into dataframes of *exactly* batchSize rows,
// except the last batch, which will be smaller if NumRows() % batchSize != 0.
// It will panic if batchSize is zero or negative.
func (df *DataFrame) SplitView(batchSize int) []*DataFrame {
  if batchSize <= 0 {
    panic("DataFrames cannot be split into batch of size batchSize<=0")
  }
  n := df.NumRows() / batchSize
  if df.NumRows() % batchSize != 0 {
    n++ // the last batch won't be full
  }
  if n <= 1 {
    return []*DataFrame{df.View()}
  }
  result := make([]*DataFrame, n)

  // create the first n-1 batches
  i := 0
  for j := 0; j < n-1; j++ {
    result[j] = df.View()
    result[j].cut(i, i+batchSize)
    i += batchSize
  }
  // last batch
  batch := df.View()
  batch.cut(i, len(df.indices))
  result[n-1] = batch

  return result
}

// SplitTrainTestViews returns a training set and a testing set.
// testingRatio is a number between 0 and 1 such that:
// testSet.NumRows() * testingRatio = df.NumRows()
// It will panic if testingRatio is not between 0 and 1 included.
// SplitTrainTestViews does not shuffle the input dataframe. It is the user's
// responsibility to shuffle the dataframe prior to splitting it.
func (df *DataFrame) SplitTrainTestViews(testingRatio float64) (*DataFrame, *DataFrame) {
  if testingRatio < 0 || testingRatio > 1 {
    panic("testing ratio cannot be below 0 or over 1")
  }
  nRows := df.NumRows()
  n := int(float64(nRows) * testingRatio)
  trainDF := df.SliceView(0, nRows-n)
  testDF := df.SliceView(nRows-n, nRows)

  return trainDF, testDF
}

// SortedView sorts the dataframe by ascending order of the given column.
// The column can either be a float, an int or a bool column.
// It will panic if the given column is neither of those.
// Missing values in integer columns will be treated as '-1'.
// If called on a bool column, it will put false values first.
// To sort in descending order, call SortedView(byColumn).ReverseView().
func (df *DataFrame) SortedView(byColumn string) *DataFrame {
  var indices []int
  if vals, ok := df.bools[byColumn]; ok {
    first := make([]int, 0, len(df.indices))
    last := make([]int, 0, len(df.indices))
    for j, i := range df.indices {
      if vals[i] {
        last = append(last, j)
      } else {
        first = append(first, j)
      }
    }
    indices = append(first, last...)
  } else {
    cpy := make([]float64, len(df.indices))
    if vals, ok := df.floats[byColumn]; ok {
      for j, i := range df.indices {
        cpy[j] = vals[i]
      }
    } else if vals, ok := df.ints[byColumn]; ok {
      for j, i := range df.indices {
        cpy[j] = float64(vals[i])
      }
    } else {
      panic(fmt.Sprintf("column %s is not a float/int/bool column", byColumn))
    }
    indices = utils.FloatArgSort(cpy, false) // readonly=false: ok to change cpy
  }
  return df.IndexView(indices)
}

// TopView returns the n rows with the lowest values if ascending=true.
// It returns the rows with the highest values if ascending=false.
// The values that serve as criteria are the values from the column byColumn.
// The column can either be a float, an int or a bool column.
// It will panic if the given column is neither of those.
// If sorted=true, rows will always be sorted according to the desired order.
// If sorted=false, rows may or may not be sorted.
// If n is higher than the total number of rows, if will return all the rows.
// It will panic if the given column is neither of those.
// Missing values in integer columns will be treated as '-1'.
// If called on a bool column, false will be treated as lower than true.
func (df *DataFrame) TopView(byColumn string, n int, ascending bool, sorted bool) *DataFrame {
  if n >= len(df.indices) {
    if !sorted {
      return df
    } else {
      n = len(df.indices)
    }
  } else if n < 0 {
    panic("n must be positive")
  }
  result := df.SortedView(byColumn)
  if ascending {
    result.indices = result.indices[:n]
  } else {
    result.indices = result.indices[len(df.indices)-n:]
    if sorted {
      result = result.ReverseView()
    }
  }
  return result
}

// ReverseView flips the order of the rows.
func (df *DataFrame) ReverseView() *DataFrame {
  result := df.View()
  size := len(df.indices)
  result.indices = make([]int, size)
  result.indexViewed = true
  for i := 0; i < size; i++ {
    result.indices[i] = df.indices[size - i - 1]
  }
  return result
}

// maxInt will vary depending on whether the code is compiled on a 32-bit or a
// 64-bit system.
const maxInt = uint64(^uint(0) >> 1)

func (inputDF *DataFrame) workerHashes(outputDF *DataFrame, columnQ utils.StringQ) {
  hash := fnv.New64a()  // TODO: benchmark this algorithm
  for col := columnQ.Next(); len(col) > 0; col = columnQ.Next() {
    defer columnQ.Notify(utils.ProcessedJob{Key: col})
    inputCol := inputDF.objects[col]
    outputCol := outputDF.ints[col]
    for _, i := range inputDF.indices {
      v := inputCol[i]
      if v == nil {
        outputCol[i] = -1 // missing value marker
      } else {
        if str, valid := v.(string); valid {
          hash.Reset()
          hash.Write([]byte(str))
          outputCol[i] = int(hash.Sum64() % maxInt)
        } else {
          outputCol[i] = -1 // a bit controversial I suppose
        }
      }
    }
  }
}

// HashStringsView hashes the string columns given as argument, thereby
// transforming string columns into integer columns.
// The hashing algorithm always returns the same int if given the same string.
// Missing strings will be converted to -1.
// This function is primarily meant to be used as a first step before
// categorical encoding.
// HashStringsView is multi-threaded.
func (df *DataFrame) HashStringsView(columns ...string) *DataFrame {
  if len(columns) == 0 {
    return df.View()
  }
  // allocate the new columns and exclude the string columns
  otherColumns := df.Header().Except(columns...).NameList()
  result := df.ColumnView(otherColumns...)
  nRows := df.NumAllocatedRows()
  for _, colName := range columns {
    result.ints[colName] = make([]int, nRows)
  }
  if len(columns) == df.NumColumns() {
    result.dataUID = generateDataUID()
  } else if len(columns) > 0 {
    result.dataUID |= generateDataUID()
  }
  // run the conversions in thread
  q := df.CreateColumnQueue(columns)
  defer q.Wait()
  for i := 0; i < q.Workers; i++ {
    go df.workerHashes(result, q)
  }
  return result
}

// DetachedView makes sure that the given columns can be altered without
// altering the original data from some parent dataframe.
// It will perform a copy only if the data is shared.
// This is useful when you execute a function that changes the data in-place:
//  view := df.DetachedView("height")
//  view.OverwriteFloats64("height", []float64{173, 174, 162, 185})
// Caveat: this can be an expensive action if the data that backs up the
// dataframe is large, even though the dataframe at hand hasn't many rows.
func (df *DataFrame) DetachedView(columns ...string) *DataFrame {
  df.debugPrint("detaching")
  result := df.View()
  result.Unshare(columns...)
  result.debugPrint("DetachedView() returns")
  return result
}

// View makes the shallowest copy of the dataframe.
// It is roughly equivalent to:
//  copy := *df
// Use this function when you want to transform an in-place operation into
// a view operation, e.g.:
//  view := df.View()
//  view.AllocateFloats("height")
func (df *DataFrame) View() *DataFrame {
  result := *df
  result.sharedMaps = true

  return &result
}

// ResetIndexView sorts the indices in order to speed up sequential access to
// the columns, including row iterators and gonum matrices.
// Speed is the only reason to reset the indices.
// It's different than pandas' eponymous function because pandas uses indices
// when concatenating columns, whereas ml-essentials does not.
// Do not use this function if order matters, as when you rely on the data being
// shuffled.
func (df *DataFrame) ResetIndexView() *DataFrame {
  result := df.View()
  result.indices = make([]int, len(df.indices))
  copy(result.indices, df.indices)
  sort.Ints(result.indices)

  return result
}
