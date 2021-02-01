package dataframe

import (
  "fmt"
  "runtime"
  "strings"
  "strconv"
  "math/rand"
  "golang.org/x/text/encoding"
  "github.com/rom1mouret/ml-essentials/v0/utils"
)

// RawData is the structure that holds the data of the dataframes.
// RawData has no concept of index view, so it always manipulates columns as
// contiguous blocks of data.
// When encapsulated in a DataBuilder, RawData allows the user to temporarily
// create columns of different size, up until it is converted to a DataFrame.
type RawData struct {
  objects   map[string][]interface{}
  floats    map[string][]float64
  bools     map[string][]bool
  ints      map[string][]int
  // columns that share memory from a parent RawData
  shared ColumnHeader
  // whether the maps objects are entirely shared
  // (not only the column data)
  sharedMaps bool
  // nil if UTF-8
  textEncoding encoding.Encoding
  // maximum number of CPUs utilized to perform operations
  // if MaxCPU <= 0 => MaxCPU = runtime.NumCPU()
  maxCPU int
  // for debugging
  structureUID uint16
  dataUID uint64
  // whether an object is a string or not
  stringHeader ColumnHeader
  // later we'll switch to:
  // special types of []interface{}. strings: 1
  // objectColTypes map[string]ObjectType
}

// ObjectType allows us to distinguish between the possible types of data
// contained in the object columns.
type ObjectType uint8
const(
  AnyObject  ObjectType = iota
  StringObject
  // TODO: more later
)

// NewRawData allocates a new RawData structure.
func NewRawData() *RawData {
  data := new(RawData)
  data.allocateEmptyMaps()
  data.SetMaxCPU(-1)
  return data
}

func (data *RawData) printableDataUID() string {
  var sb strings.Builder
  v := data.dataUID
  for i := 0; v > 0; i++ {
    if v % 2 == 1 {
      sb.WriteString("|")
      sb.WriteString(strconv.FormatInt(int64(i), 36))
    }
    v /= 2
  }
  sb.WriteString("|")
  return sb.String()
}

func (data *RawData) resetStructureUID() {
  data.structureUID = uint16(rand.Uint32() % 65536)
}

func generateDataUID() uint64 {
  bitPosition := rand.Uint32() % 64
  return 1 << bitPosition
}

// NumAllocatedRows returns the total number of allocated rows.
// If this is called via a pointer on a dataframe, the number of allocated rows
// can be different than the value returned by NumRows().
// This method is not reliable on a RawData under construction, e.g. when its
// columns are being built via a DataBuilder.
func (data *RawData) NumAllocatedRows() int {
  for _, vals := range data.objects {
    return len(vals)
  }
  for _, vals := range data.floats {
    return len(vals)
  }
  for _, vals := range data.ints {
    return len(vals)
  }
  for _, vals := range data.bools {
    return len(vals)
  }
  return 0
}

// ToDataFrame upgrades the RawData structure to a DataFrame.
// It will panic if the columns are of different size.
// The data is shared between the original RawData and the returned dataframe,
// so any change to the RawData will affect the dataframe, and vice versa.
func (data *RawData) ToDataFrame() *DataFrame {
  // expected number of rows
  nRows := data.NumAllocatedRows()

  // consistency check
  for col, vals := range data.objects {
    if nRows != len(vals) {
      panic(fmt.Sprintf("object column %s has %d rows. Expected: %d", col, len(vals), nRows))
    }
  }
  for col, vals := range data.floats {
    if nRows != len(vals) {
      panic(fmt.Sprintf("float column %s has %d rows. Expected: %d", col, len(vals), nRows))
    }
  }
  for col, vals := range data.bools {
    if nRows != len(vals) {
      panic(fmt.Sprintf("bool column %s has %d rows. Expected: %d", col, len(vals), nRows))
    }
  }
  for col, vals := range data.ints {
    if nRows != len(vals) {
      panic(fmt.Sprintf("int column %s has %d rows. Expected: %d", col, len(vals), nRows))
    }
  }
  result := new(DataFrame)
  result.objects = data.objects
  result.floats = data.floats
  result.bools = data.bools
  result.ints = data.ints
  result.maxCPU = data.maxCPU
  result.stringHeader = data.stringHeader
  result.mask = make([]bool, nRows)
  result.indices = utils.MakeRange(0, nRows, 1)
  result.dataUID = data.dataUID
  result.textEncoding = data.textEncoding
  result.resetStructureUID()

  return result
}

// NumColumns returns the total number of columns.
func (data *RawData) NumColumns() int {
  return len(data.ints) + len(data.floats) + len(data.bools) + len(data.objects)
}

// SetMaxCPU sets the number of CPUs that are allowed to be utilized by the
// functions operating on the dataframe and any view on the dataframe.
// If maxCPU is 0 or negative, Max CPU will be set to the number of CPU cores
// on your machine.
func (data *RawData) SetMaxCPU(maxCPU int) {
  nCPU := runtime.NumCPU()
  if maxCPU <= 0 || maxCPU >= nCPU {
    data.maxCPU = nCPU
  } else {
    data.maxCPU = maxCPU
  }
}

// ActualMaxCPU returns the maximum number of CPUs that are allowed to be
// utilized by the functions operating on the dataframe.
// If such a maximum number of CPUs was never set or or if it was set with a
// number higher than the number of CPU cores on your machine, it will return
// the number of CPU cores on your machine.
func (data *RawData) ActualMaxCPU() int {
  return data.maxCPU
}

// Unshare is the in-place, low-level version of DataFrame.DetachView().
// For your own sake, please use DataFrame.DetachView() instead.
func (data *RawData) Unshare(columns...string) {
  // deep-copy the columns that are viewed
  if !data.sharedMaps && data.shared.Num() == 0 {
    return
  }
  if data.sharedMaps {
    data.reallocateMaps()
  }
  var colSet map[string]bool
  if len(columns) == 0 || len(columns) == data.NumColumns() {
    colSet = data.Header().NameSet()
    data.dataUID = generateDataUID()
  } else {
    colSet = utils.ToStringSet(columns)
    data.dataUID |= generateDataUID()
  }
  for col, v := range data.objects {
    if data.shared.contains(col) && colSet[col] {
      series := make([]interface{}, len(v))
      copy(series, v)
      data.objects[col] = series
    }
  }
  for col, v := range data.floats {
    if data.shared.contains(col) && colSet[col] {
      series := make([]float64, len(v))
      copy(series, v)
      data.floats[col] = series
    }
  }
  for col, v := range data.ints {
    if data.shared.contains(col) && colSet[col] {
      series := make([]int, len(v))
      copy(series, v)
      data.ints[col] = series
    }
  }
  for col, v := range data.bools {
    if data.shared.contains(col) && colSet[col] {
      series := make([]bool, len(v))
      copy(series, v)
      data.bools[col] = series
    }
  }
  for col := range colSet {
    data.shared.remove(col)
  }
}

// Drop removes the given columns.
func (data *RawData) Drop(columns ...string) {
  if len(columns) == 0 {
    return
  }
  for _, col := range columns {
    data.shared.remove(col)
    data.stringHeader.remove(col)
    delete(data.ints, col)
    delete(data.bools, col)
    delete(data.floats, col)
    delete(data.objects, col)
  }
}

// Rename changes the name of a column.
// The new column will be of the same type and share the same data.
// For example, if you execute:
//  df.View().Rename("apples", "oranges").Ints("oranges").Set(0, 42)
// It will change df's number of apples to 42 at index=0.
func (data *RawData) Rename(oldName string, newName string) {
  if data.sharedMaps {
    data.reallocateMaps()
  }
  if vals, ok := data.floats[oldName]; ok {
    data.floats[newName] = vals
    delete(data.floats, oldName)
  } else if vals, ok := data.objects[oldName]; ok {
    data.objects[newName] = vals
    delete(data.objects, oldName)
    if data.stringHeader.contains(oldName) {
      data.stringHeader.remove(oldName)
      data.stringHeader.add(newName)
    }
  } else if vals, ok := data.ints[oldName]; ok {
    data.ints[newName] = vals
    delete(data.ints, oldName)
  } else if vals, ok := data.bools[oldName]; ok {
    data.bools[newName] = vals
    delete(data.bools, oldName)
  } else {
    panic(fmt.Sprintf("%s is not a column", oldName))
  }
  if data.shared.contains(oldName) {
    data.shared.remove(oldName)
    data.shared.add(newName)
  }
}

// AllocInts allocates new empty integer columns.
func (data *RawData) AllocInts(columns ...string) {
  if data.sharedMaps && len(columns) > 0 {
    data.reallocateMaps()
  }
  nRows := data.NumAllocatedRows()
  for _, col := range columns {
    data.ints[col] = make([]int, nRows)
  }
}

// AllocFloats allocates new empty float columns.
func (data *RawData) AllocFloats(columns ...string) {
  if data.sharedMaps && len(columns) > 0 {
    data.reallocateMaps()
  }
  nRows := data.NumAllocatedRows()
  for _, col := range columns {
    data.floats[col] = make([]float64, nRows)
  }
}

// AllocBools allocates new empty float columns.
func (data *RawData) AllocBools(columns ...string) {
  if data.sharedMaps && len(columns) > 0 {
    data.reallocateMaps()
  }
  nRows := data.NumAllocatedRows()
  for _, col := range columns {
    data.bools[col] = make([]bool, nRows)
  }
}

// AllocObjects allocates new empty object columns.
func (data *RawData) AllocObjects(columns ...string) {
  if data.sharedMaps && len(columns) > 0 {
    data.reallocateMaps()
  }
  nRows := data.NumAllocatedRows()
  for _, col := range columns {
    data.objects[col] = make([]interface{}, nRows)
  }
}

// AllocStrings allocates new empty string columns.
func (data *RawData) AllocStrings(columns ...string) {
  if data.sharedMaps && len(columns) > 0 {
    data.reallocateMaps()
  }
  data.AllocObjects(columns...)
  for _, col := range columns {
    data.stringHeader.add(col)
  }
}

// TransferRawDataFrom adds the data from another RawData structure.
// The two structures will share data, so changing one will change the other.
// Reminder: RawData's functions are ignorant of dataframe indices, so don't
// expect this function to exlusively transfer viewed data when it's called on a
// dataframe.
func (data *RawData) TransferRawDataFrom(from *RawData) {
  if data.sharedMaps {
    data.reallocateMaps()
  }
  if data.NumColumns() == 0 {
    data.dataUID = from.dataUID
  } else {
    data.dataUID |= from.dataUID
  }
  for col, vals := range from.bools {
    data.bools[col] = vals
    data.shared.add(col)
  }
  for col, vals := range from.ints {
    data.ints[col] = vals
    data.shared.add(col)
  }
  for col, vals := range from.floats {
    data.floats[col] = vals
    data.shared.add(col)
  }
  for col, vals := range from.objects {
    data.objects[col] = vals
    data.shared.add(col)
  }
  for col := range from.stringHeader.get() {
    data.stringHeader.add(col)
  }
}

// MergeRawDataColumns transfers data from multiple RawData structures.
// It basically calls TransferRawDataFrom on each RawData passed as argument and
// it is subject to the same limitations.
func MergeRawDataColumns(list []*RawData) *RawData {
  result := NewRawData()
  for _, element := range list {
    result.TransferRawDataFrom(element)
    result.textEncoding = element.textEncoding
    result.maxCPU = element.maxCPU
  }
  return result
}

// MergeRawDataRows concatenates multiple RawData together in a row-wise manner.
// RawData can have different columns (not recommended), but be aware that it
// will panic if you try right away to uprade the resulting RawData to a
// dataframe.
// The data will be copied and the given structures won't share data, so
// altering one of the RawData later won't affect the returned RawData.
// It is the equivalent of numpy.concat(list, axis=0)
func MergeRawDataRows(list []*RawData) *RawData {
  result := NewRawData()
  if len(list) == 0 {
    return result
  }
  // pre-allocate the rows to avoid new allocations by append(arr, ...values)
  size := 0
  for _, data := range list {
    size += data.NumAllocatedRows()
  }
  for col := range list[0].floats {
    result.floats[col] = make([]float64, 0, size)
  }
  for col := range list[0].bools {
    result.bools[col] = make([]bool, 0, size)
  }
  for col := range list[0].ints {
    result.ints[col] = make([]int, 0, size)
  }
  for col := range list[0].objects {
    result.objects[col] = make([]interface{}, 0, size)
  }
  // copy the data with append()
  // (https://gist.github.com/xogeny/b819af6a0cf8ba1caaef)
  for _, data := range list {
    for col, vals := range data.floats {
      result.floats[col] = append(result.floats[col], vals...)
    }
    for col, vals := range data.bools {
      result.bools[col] = append(result.bools[col], vals...)
    }
    for col, vals := range data.ints {
      result.ints[col] = append(result.ints[col], vals...)
    }
    for col, vals := range data.objects {
      result.objects[col] = append(result.objects[col], vals...)
    }
  }
  return result
}

// IntToFloats converts integer columns into float columns.
// Note that this runs as at a view-free level since it wouldn't make sense to
// convert only parts of dataframe's column, given that mixed types are not
// allowed for numerical columns.
func (data *RawData) IntToFloats(columns...string) {
  if data.sharedMaps {
    data.reallocateMaps()
  }
  for _, col := range columns {
    olddata := data.ints[col]
    newdata := make([]float64, len(olddata))
    for i, v := range olddata {
      newdata[i] = float64(v)
    }
    delete(data.ints, col)
    data.floats[col] = newdata
  }
}

// BoolToFloats converts boolean columns into float columns.
// Note that this runs as at a view-free level since it wouldn't make sense to
// convert only parts of dataframe's column, given that mixed types are not
// allowed for numerical columns.
func (data *RawData) BoolToFloats(columns...string) {
  if data.sharedMaps {
    data.reallocateMaps()
  }
  for _, col := range columns {
    olddata := data.bools[col]
    newdata := make([]float64, len(olddata))
    for i, v := range olddata {
      if v {
        newdata[i] = 1.0
      }
    }
    delete(data.ints, col)
    data.floats[col] = newdata
  }
}

// Only used by other ml-essentials' packages.
func (data *RawData) CreateColumnQueue(columns []string) utils.StringQ {
  return utils.CreateStringQueue(columns, data.ActualMaxCPU())
}

// this basically transforms a view with:
// .sharedMaps = true
// -- to -->
// .shared = {all cols}
// .sharedMaps = false
func (data *RawData) reallocateMaps() {
  tmp := NewRawData()
  tmp.TransferRawDataFrom(data) // put everything in shared
  data.bools = tmp.bools
  data.ints = tmp.ints
  data.floats = tmp.floats
  data.objects = tmp.objects
  data.shared = tmp.shared  // all columns
  data.sharedMaps = false
}

func (data *RawData) allocateEmptyMaps() {
  data.floats = make(map[string][]float64)
  data.objects = make(map[string][]interface{})
  data.bools = make(map[string][]bool)
  data.ints = make(map[string][]int)
  data.dataUID = generateDataUID()
  data.sharedMaps = false
  data.stringHeader = ColumnHeader{}
  data.resetStructureUID()
}

func (df *RawData) PrintUIDs() {
  fmt.Printf("structure UID: %d; data UID: %s\n",
             df.structureUID, df.printableDataUID())
}
