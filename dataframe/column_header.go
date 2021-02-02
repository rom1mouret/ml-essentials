package dataframe

// ColumnHeader helps you manipulate column names
type ColumnHeader struct {
  columns map[string]bool
}

// Columns create a ColumnHeader from a list of column names.
func Columns(names ...string) ColumnHeader {
  result := make(map[string]bool)
  for _, col := range names {
    result[col] = true
  }
  return ColumnHeader{result}
}

// Num returns the number of columns in the ColumnHeader
func (h ColumnHeader) Num() int {
  if h.columns == nil {
    return 0
  }
  return len(h.columns)
}

// NameSet returns the set of columns in the header for read-only access.
// This is faster than NameList()
func (h ColumnHeader) NameSet() map[string]bool {
  if h.columns == nil {
    return make(map[string]bool)
  }
  return h.columns
}

// NameList returns the list of columns in the header.
// Altering the returned slice won't alter ColumnHeader.
func (h ColumnHeader) NameList() []string {
  if h.columns == nil {
    return nil
  }
  result := make([]string, len(h.columns))
  i := 0
  for col := range h.columns {
    result[i] = col
    i++
  }
  return result
}

// Except removes all columns from other ColumnHeaders.
// It returns a shallow-copy of itself, not an entirely new ColumnHeader.
func (h ColumnHeader) ExceptHeader(others ...ColumnHeader) ColumnHeader {
  if h.columns == nil {
    return h
  }
  for _, other := range others {
    if other.columns != nil {
      for col := range other.columns {
        delete(h.columns, col)
      }
    }
  }
  return h
}

// Except removes all the columns given as arguments.
// It returns a shallow-copy of itself, not an entirely new ColumnHeader.
func (h ColumnHeader) Except(columns ...string) ColumnHeader {
  if h.columns == nil {
    return h
  }
  for _, colName := range columns {
    delete(h.columns, colName)
  }
  return h
}

// And add all the columns from the given other ColumnHeaders.
// It returns a shallow-copy of itself, not an entirely new ColumnHeader.
func (h ColumnHeader) And(others ...ColumnHeader) ColumnHeader {
  if h.columns == nil {
    h.columns = make(map[string]bool)
  }
  for _, other := range others {
    if other.columns != nil {
      for col := range other.columns {
        h.columns[col] = true
      }
    }
  }
  return h
}

// Copy returns a deep-copy of the ColumnHeader
func (h ColumnHeader) Copy() ColumnHeader {
  if h.columns == nil {
    return h
  }
  result := make(map[string]bool)
  if h.columns != nil {
    for col := range h.columns {
      result[col] = true
    }
  }
  return ColumnHeader{result}
}

// IntHeader returns a ColumnHeader with all the integer column names
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) IntHeader() ColumnHeader {
  if len(data.ints) == 0 {
    return ColumnHeader{}
  }
  result := make(map[string]bool)
  for col := range data.ints {
    result[col] = true
  }
  return ColumnHeader{result}
}

// ObjectHeader returns a ColumnHeader with all the object column names,
// including that of string columns.
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) ObjectHeader() ColumnHeader {
  if len(data.objects) == 0 {
    return ColumnHeader{}
  }
  result := make(map[string]bool)
  for col := range data.objects {
    result[col] = true
  }
  return ColumnHeader{result}
}

// FloatHeader returns a ColumnHeader with all the float column names.
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) FloatHeader() ColumnHeader {
  if len(data.floats) == 0 {
    return ColumnHeader{}
  }
  result := make(map[string]bool)
  for col := range data.floats {
    result[col] = true
  }
  return ColumnHeader{result}
}

// BoolHeader returns a ColumnHeader with all the boolean column names.
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) BoolHeader() ColumnHeader {
  if len(data.bools) == 0 {
    return ColumnHeader{}
  }
  result := make(map[string]bool)
  for col := range data.bools {
    result[col] = true
  }
  return ColumnHeader{result}
}

// BoolHeader returns a ColumnHeader with all the string column names.
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) StringHeader() ColumnHeader {
  // we do a copy for consistency with other Getters
  return data.stringHeader.Copy()
}

// Header returns a ColumnHeader with all the column names.
// Altering the returned ColumnHeader has no effect on the underlying RawData.
func (data *RawData) Header() ColumnHeader {
  return data.IntHeader().And(data.BoolHeader(), data.ObjectHeader(), data.FloatHeader())
}

func (h *ColumnHeader) add(cols ...string) {
  if h.columns == nil {
    h.columns = make(map[string]bool)
  }
  for _, col := range cols {
    h.columns[col] = true
  }
}

func (h ColumnHeader) remove(cols ...string) {
  if h.columns == nil {
    return
  }
  for _, col := range cols {
    delete(h.columns, col)
  }
}

func (h ColumnHeader) contains(col string) bool {
  if h.columns == nil {
    return false
  }
  return h.columns[col]
}

func (h ColumnHeader) get() map[string]bool {
  if h.columns == nil {
    return make(map[string]bool)
  }
  return h.columns
}
