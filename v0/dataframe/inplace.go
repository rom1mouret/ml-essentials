package dataframe

import (
  "golang.org/x/text/encoding"
)

func inplaceEncoding(values []interface{}, indices []int, decoder *encoding.Decoder,
                     encoder *encoding.Encoder) (error, int) {
  for j, i := range indices {  // others are left to nil
    if values[i] != nil {
      var err error
      v := values[i].(string)
      // 1. convert to UTF-8 if not already UTF-8
      if decoder != nil {
        v, err = decoder.String(v)
        if err != nil {
          return err, j
        }
      } // otherwise UTF-8
      // 2. convert from UTF-8 to new encoding
      if encoder != nil {
        v, err = encoder.String(v)
        if err != nil {
          return err, j
        }
      } // otherwise UTF-8
      // 3. set the new value
      values[i] = v
    }
  }
  return nil, -1
}

// Encode changes the encoding of all all the string columns.
// It returns an error if it cannot be encoded into the desired encoding, or
// decoded using the current encoding.
// If encoding is nil, strings will be encoded in UTF-8.
func (df *DataFrame) Encode(newEncoding encoding.Encoding) error {
  // TODO: multi-thread this function
  df.debugPrint("in-place encoding on")
  if df.textEncoding == newEncoding {
    return nil
  }
  var decoder *encoding.Decoder
  var encoder *encoding.Encoder
  if df.textEncoding != nil {
    decoder = df.textEncoding.NewDecoder()
  }
  if newEncoding != nil {
    encoder = newEncoding.NewEncoder()
  }
  colNames := df.stringHeader.NameList()
  for i, col := range colNames {
    values := df.objects[col]
    err, where := inplaceEncoding(values, df.indices, decoder, encoder)
    if err != nil {
      // we need to revert our changes
      if newEncoding == nil {
        decoder = nil
      } else {
        decoder = newEncoding.NewDecoder()
      }
      if df.textEncoding == nil {
        encoder = nil
      } else {
        encoder = df.textEncoding.NewEncoder()
      }
      inplaceEncoding(values, df.indices[:where], decoder, encoder)
      for _, previous := range colNames[:i] {
          inplaceEncoding(df.objects[previous], df.indices, decoder, encoder)
      }
      return err
    }
  }
  df.textEncoding = newEncoding

  return nil
}

// OverwriteInts (over)writes the given column with the given values.
// The given slice is copied, so it can safely be alteredafter this call.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Ints(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, values[i])
//  }
func (df *DataFrame) OverwriteInts(colName string, values []int) {
  df.debugPrint("overwriting ints on")
  col := df.ints[colName]
  if len(col) == 0 {
    df.AllocInts(colName)
    col = df.ints[colName]
  }
  for j, i := range  df.indices {
    col[i] = values[j]
  }
}

// OverwriteFloats64 (over)writes the given column with the given values.
// The given slice is copied, so it can safely be altered after this call.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Floats(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, values[i])
//  }
func (df *DataFrame) OverwriteFloats64(colName string, values []float64) {
  df.debugPrint("overwriting floats64 on")
  col := df.floats[colName]
  if len(col) == 0 {
    df.AllocFloats(colName)
    col = df.floats[colName]
  }
  for j, i := range df.indices {
    col[i] = values[j]
  }
}

// OverwriteFloats32 (over)writes the given column with the given values.
// The given slice is copied, so it can safely be altered after this call.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Floats(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, (float64) values[i])
//  }
func (df *DataFrame) OverwriteFloats32(colName string, values []float32) {
  df.debugPrint("overwriting floats32 on")
  col := df.floats[colName]
  if len(col) == 0 {
    df.AllocFloats(colName)
    col = df.floats[colName]
  }
  for j, i := range df.indices {
    col[i] = float64(values[j])
  }
}

// OverwriteBools (over)writes the given column with the given values.
// The given slice is copied, so it can safely be altered after this call.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Bools(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, values[i])
//  }
func (df *DataFrame) OverwriteBools(colName string, values []bool) {
  df.debugPrint("overwriting bools on")
  col := df.bools[colName]
  if len(col) == 0 {
    df.AllocBools(colName)
    col = df.bools[colName]
  }
  for j, i := range df.indices {
    col[i] = values[j]
  }
}

// OverwriteObjects (over)writes the given column with the given values.
// The given slice is copied, so it can safely be altered after this call.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Objects(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, values[i])
//  }
// The third argument is only used if the column doesn't exist and has to be
// created. It is the only way to mix strings with nil values and yet benefit
// from dataframe operations specialized for strings such as HashStringsView.
func (df *DataFrame) OverwriteObjects(colName string, values []interface{},
                                      objectType ObjectType) {
  df.debugPrint("overwriting objects on")
  col := df.objects[colName]
  if len(col) == 0 {
    if objectType == 1 {
      df.AllocStrings(colName)
    } else {
      df.AllocObjects(colName)
    }
    col = df.objects[colName]
  }
  for j, i := range df.indices {
    col[i] = values[j]
  }
}

// OverwriteStrings (over)writes the given column with the given values.
// The given values are always copied, so the given slice can be safely altered
// after calling this function.
// If the column doesn't exist, it will create a new column.
// Otherwise, it is functionally equivalent to:
//  access := df.Objects(colName)
//  for i := 0; i < len(values); i++ {
//    access.Set(i, values[i])
//  }
// If you need to overwrite strings with missing values, use OverwriteObjects
// instead.
func (df *DataFrame) OverwriteStrings(colName string, values []string) {
  df.debugPrint("overwriting strings on")
  col := df.objects[colName]
  if len(col) == 0 {
    df.AllocStrings(colName)
    col = df.objects[colName]
  }
  for j, i := range df.indices {
    col[i] = values[j]
  }
}

func (df *DataFrame) cut(from int, to int) *DataFrame {
  // heuristic to guess when the maps are worth reallocating
  df.debugPrint("cutting")
  worthReallocating := (to - from) > 10 * df.NumColumns()
  if df.indexViewed || (df.sharedMaps && !worthReallocating) {
    df.indices = df.indices[from:to]
    df.indexViewed = true
  } else {
    // this is going to be faster downstream
    if df.sharedMaps {
      df.reallocateMaps()
    }
    df.indices = df.indices[:to - from]  // same as range(0, to-from)
    for col, vals := range df.floats {
      df.floats[col] = vals[from:to]
      df.shared.add(col)
    }
    for col, vals := range df.ints {
      df.ints[col] = vals[from:to]
      df.shared.add(col)
    }
    for col, vals := range df.bools {
      df.bools[col] = vals[from:to]
      df.shared.add(col)
    }
    for col, vals := range df.objects {
      df.objects[col] = vals[from:to]
      df.shared.add(col)
    }
  }
  df.debugPrint("cut() returns")
  return df
}
