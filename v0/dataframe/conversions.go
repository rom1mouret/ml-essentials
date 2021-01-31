package dataframe

import (
  "fmt"
)

// CopyValuesToInterfaces returns a copy of a column's data packed into an
// interface slice, regardless of the column's type.
func (df *DataFrame) CopyValuesToInterfaces(colName string) []interface{} {
  result := make([]interface{}, df.NumRows())
  var copied bool
  if vals, ok := df.objects[colName]; ok {
    for j, i := range df.indices {
      result[j] = vals[i]
    }
    copied = true
  } else if vals, ok := df.floats[colName]; ok {
    for j, i := range df.indices {
      result[j] = vals[i]
    }
    copied = true
  } else if vals, ok := df.ints[colName]; ok {
    for j, i := range df.indices {
      result[j] = vals[i]
    }
    copied = true
  } else if vals, ok := df.bools[colName]; ok {
    for j, i := range df.indices {
      result[j] = vals[i]
    }
    copied = true
  }
  if !copied {
    panic(fmt.Sprintf("column %s is not in the dataframe", colName))
  }
  return result
}

// LabelToInt maps one column's string values to a range of integers starting
// from zero, and returns both the converted strings and the mapping.
// For example: conversion(['a', 'b', 'a', 'c']) -> [0, 1, 0, 2] via the mapping
// a: 0, b: 1, c: 2.
// If a string is nil, the string will be converted to -1.
// Use this column to convert classification labels into integers.
func (df *DataFrame) LabelToInt(colName string) ([]int, map[string]int) {
  if !df.stringHeader.NameSet()[colName] {
    panic(fmt.Sprintf("%s is not in the set of string columns", colName))
  }
  vals := df.objects[colName]
  mapping := make(map[string]int)
  result := make([]int, len(df.indices))
  for j, i := range df.indices {
    v := vals[i]
    if v == nil {
      result[j] = -1
    } else {
      str := v.(string)
      if ordinal, known := mapping[str]; known {
        result[j] = ordinal
      } else {
        result[j] = len(mapping)
        mapping[str] = result[j]
      }
    }
  }
  return result, mapping
}
