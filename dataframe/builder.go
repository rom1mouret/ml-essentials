package dataframe

import "golang.org/x/text/encoding"

// DataBuilder is a helper structure to build dataframes.
// Use dataframe.DataBuilder{RawData: dataframe.EmptyRawData()} to initialize it
type DataBuilder struct {
  RawData *RawData
}

// AddFloats adds a list of floats to the given float column.
// It returns a shallow copy of itself.
func (builder DataBuilder) AddFloats(col string, values ...float64) DataBuilder {
  builder.RawData.floats[col] = append(builder.RawData.floats[col], values...)
  return builder
}

// AddFloats adds a list of bools to the given boolean column.
// It returns a shallow copy of itself.
func (builder DataBuilder) AddBools(col string, values ...bool) DataBuilder {
  builder.RawData.bools[col] = append(builder.RawData.bools[col], values...)
  return builder
}

// AddInts adds a list of ints to the given int column.
// It returns a shallow copy of itself.
func (builder DataBuilder) AddInts(col string, values ...int) DataBuilder {
  builder.RawData.ints[col] = append(builder.RawData.ints[col], values...)
  return builder
}

// AddObjects adds a list of objects to the given object column.
// It returns a shallow copy of itself.
// You can use this function to add strings too.
func (builder DataBuilder) AddObjects(col string, values ...interface{}) DataBuilder {
  builder.RawData.objects[col] = append(builder.RawData.objects[col], values...)
  return builder
}

// MarkAsString tags a given object column as a string-only column.
// This gives access to functionalities that generic object columns don't have.
func (builder DataBuilder) MarkAsString(col string) DataBuilder {
  builder.RawData.stringHeader.add(col)
  return builder
}

// AddStrings adds a list of strings to the given object column.
// It returns a shallow copy of itself.
// If you need to add nils (= missing value), use AddObjects(col, ...)
// followded by MarkAsString(col).
func (builder DataBuilder) AddStrings(col string, values ...string) DataBuilder {
  interfaces := make([]interface{}, len(values))
  for i, v := range values {
    interfaces[i] = v
  }
  builder.AddObjects(col, interfaces...)
  builder.MarkAsString(col)

  return builder
}

// SetFloats adds or replaces the values of the given float column.
// Values are not copied, so if you change them it will change them everywhere.
// It returns a shallow copy of itself.
func (builder DataBuilder) SetFloats(col string, values []float64) DataBuilder {
  builder.RawData.floats[col] = values
  return builder
}

// SetBools adds or replaces the values of the given boolean column.
// Values are not copied, so if you change them it will change them everywhere.
// It returns a shallow copy of itself.
func (builder DataBuilder) SetBools(col string, values []bool) DataBuilder {
  builder.RawData.bools[col] = values
  return builder
}

// SetInts adds or replaces the values of the given integer column.
// Values are not copied, so if you change them it will change them everywhere.
// It returns a shallow copy of itself.
func (builder DataBuilder) SetInts(col string, values []int) DataBuilder {
  builder.RawData.ints[col] = values
  return builder
}

// SetObjects adds or replaces the values of the given object column.
// Values are not copied, so if you change them it will change them everywhere.
// It returns a shallow copy of itself.
// If you want to set a slice of strings, you'll need to convert the slice to a
// slice of interfaces and call MarkAsString(col).
func (builder DataBuilder) SetObjects(col string, values []interface{}) DataBuilder {
  builder.RawData.objects[col] = values
  return builder
}

// TextEncoding informs ml-essential that the strings that you have provided
// are encoded in the given encoding.
// If this function is never called or if nil is passed as argument, it will be
// assumed that all the strings are utf8-encoded.
// Even if the strings are not utf8-encoded, it is not mandatory to call this
// function since encoding is rarely ever used by ml-essentials.
// TextEncoding returns a shallow copy of itself.
func (builder DataBuilder)  TextEncoding(encoding encoding.Encoding) DataBuilder {
  builder.RawData.textEncoding = encoding
  return builder
}

// ToDataFrame() creates a dataframe out of the RawData object.
// It will panic if the columns are of different size.
// The returned dataframe shares its data and structure with the encapsulated
// rawdata.
func (builder DataBuilder) ToDataFrame() *DataFrame {
  return builder.RawData.ToDataFrame()
}

/// THIS IS ONLY FOR TESTING

func fillBlanks(builder DataBuilder) *DataFrame {
  data := builder.RawData
  rows := data.NumAllocatedRows()
  if len(data.objects) == data.stringHeader.Num() {
    arr := make([]interface{}, rows)
    for i := range arr {
      if i % 2 == 0 {
        arr[i] = make([]bool, 1)
      }
    }
    builder.AddObjects("SomeObjects", arr...)
  }
  if data.stringHeader.Num() == 0 {
    arr := make([]string, rows)
    for i := range arr {
      if i % 2 == 0 {
        arr[i] = "one"
      } else {
        arr[i] = "two"
      }
    }
    builder.AddStrings("SomeStrings", arr...)
  }
  if len(data.ints) == 0 {
    arr := make([]int, rows)
    for i := range arr {
      arr[i] = i
    }
    builder.AddInts("SomeInts", arr...)
  }
  if len(data.bools) == 0 {
    arr := make([]bool, rows)
    for i := range arr {
      arr[i] = i % 2 == 0
    }
    builder.AddBools("SomeBools", arr...)
  }
  if len(data.floats) == 0 {
    arr := make([]float64, rows)
    for i := range arr {
      arr[i] = float64(i) / float64(rows)
    }
    builder.AddFloats("SomeFloats", arr...)
  }
  return data.ToDataFrame()
}
