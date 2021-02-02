package dataframe

import (
  "fmt"
  "gonum.org/v1/gonum/mat"
)

type ColumnAccess struct {
  indices     []int
  contiguous  bool
}

// FloatAccess is a random-access iterator for float columns.
type FloatAccess struct {
  ColumnAccess
  rawData []float64
}

// IntAccess is a random-access iterator for integer columns.
type IntAccess struct {
  ColumnAccess
  rawData []int
}

// BoolAccess is a random-access iterator for boolean columns.
type BoolAccess struct {
  ColumnAccess
  rawData []bool
}

// ObjectAccess is a random-access iterator for object columns, including
// string columns.
type ObjectAccess struct {
  ColumnAccess
  rawData []interface{}
}

// StringAccess is a random-access iterator for string columns
type StringAccess struct {
  ColumnAccess
  rawData []interface{}
}

// Ints returns an iterator on a given integer column
func (df *DataFrame) Ints(colName string) IntAccess {
  df.debugPrint("int column access")
  if data, ok := df.ints[colName]; ok {
    return IntAccess{
        ColumnAccess: ColumnAccess{indices: df.indices},
        rawData: data}
  } else {
    panic(fmt.Sprintf("%s is not in the list of int columns", colName))
  }
}

// Bools returns an iterator on a given boolean column
func (df *DataFrame) Bools(colName string) BoolAccess {
  df.debugPrint("bool column access")
  if data, ok := df.bools[colName]; ok {
    return BoolAccess{
        ColumnAccess: ColumnAccess{indices: df.indices},
        rawData: data}
  } else {
    panic(fmt.Sprintf("%s is not in the list of bool columns", colName))
  }
}

// Floats returns an iterator on a given float column
func (df *DataFrame) Floats(colName string) FloatAccess {
  df.debugPrint("float column access")
  if data, ok := df.floats[colName]; ok {
    return FloatAccess{
        ColumnAccess: ColumnAccess{indices: df.indices,
                                   contiguous: !df.indexViewed},
        rawData: data}
  } else {
    panic(fmt.Sprintf("%s is not in the list of float columns", colName))
  }
}

// Objects returns an iterator on a given object column, including string
// columns.
func (df *DataFrame) Objects(colName string) ObjectAccess {
  df.debugPrint("object column access")
  if data, ok := df.objects[colName]; ok {
    return ObjectAccess{
        ColumnAccess: ColumnAccess{indices: df.indices},
        rawData: data}
  } else {
    panic(fmt.Sprintf("%s is not in the list of object columns", colName))
  }
}

// Strings returns an iterator on a given string column.
func (df *DataFrame) Strings(colName string) StringAccess {
  df.debugPrint("string column access")
  if data, ok := df.objects[colName]; ok {
    if !df.stringHeader.contains(colName) {
      panic(fmt.Sprintf("%s is an object column but not marked as string", colName))
    }
    return StringAccess{
        ColumnAccess: ColumnAccess{indices: df.indices},
        rawData: data}
  } else {
    panic(fmt.Sprintf("%s is not in the list of object columns", colName))
  }
}

// Size returns the length of the column.
func (access ColumnAccess) Size() int {
  return len(access.indices)
}

// SharedIndex returns the index to the backing data, given a index to the
// column.
// You probably don't need this function.
func (access ColumnAccess) SharedIndex(localIndex int) int {
  return access.indices[localIndex]
}

// VecDense creates a gonum's VecDense object from the dataframe's float data.
// It always copies the data, so you can change the returned VecDense without
// changing the dataframe.
func (access FloatAccess) VecDenseCopy() *mat.VecDense {
  data := make([]float64, len(access.indices))
  for j, i := range access.indices {
    data[j] = access.rawData[i]
  }
  return mat.NewVecDense(len(data), data)
}

// VecDense creates a gonum's VecDense object from the dataframe's float data.
// The data is not copied if the underlying dataframe is contiguous, otherwise
// the data is copied.
// Use this function if you are not going to change the returned VecDense and
// want to avoid an unnecessary copy.
func (access FloatAccess) VecDense() *mat.VecDense {
  if access.contiguous {
    return mat.NewVecDense(len(access.rawData), access.rawData)
  }
  return access.VecDenseCopy()
}

// TODO: add VecDense for bools and ints

// Get returns the float value at the given index.
func (access FloatAccess) Get(row int) float64 {
  return access.rawData[access.indices[row]]
}

// Set overwrites the float value at the given index.
func (access FloatAccess) Set(row int, val float64) {
  access.rawData[access.indices[row]] = val
}

// Get returns the integer at the given index.
func (access IntAccess) Get(row int) int {
  return access.rawData[access.indices[row]]
}

// Set overwrites the integer value at the given index.
func (access IntAccess) Set(row int, val int) {
  access.rawData[access.indices[row]] = val
}

// Get returns the boolean value at the given index.
func (access BoolAccess) Get(row int) bool {
  return access.rawData[access.indices[row]]
}

// Set overwrites the boolean value at the given index.
func (access BoolAccess) Set(row int, val bool) {
  access.rawData[access.indices[row]] = val
}

// Get returns the object at the given index.
func (access ObjectAccess) Get(row int) interface{} {
  return access.rawData[access.indices[row]]
}

// Set overwrites the object at the given index.
func (access ObjectAccess) Set(row int, val interface{}) {
  access.rawData[access.indices[row]] = val
}

// Get returns the string at the given index.
func (access StringAccess) Get(row int) string {
  return access.rawData[access.indices[row]].(string)
}

// Set overwrites the string at the given index.
func (access StringAccess) Set(row int, val string) {
  access.rawData[access.indices[row]] = val
}
