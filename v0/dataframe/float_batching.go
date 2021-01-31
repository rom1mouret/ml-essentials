package dataframe

import (
  "fmt"
)

type FloatBatching struct {
  bColumns    []int
  iColumns    []int
  fColumns    []int
  columns     []string
  initialized bool
}

func imin(x, y int) int {
  if x < y {
    return x
  }
  return y
}

func (bat *FloatBatching) initialize(df *DataFrame, columns []string) {
  if !bat.initialized {
    bat.initialized = true
    bat.columns = columns
    bat.fColumns = make([]int, 0, imin(len(df.floats), len(bat.columns)))
    bat.bColumns = make([]int, 0, imin(len(df.bools), len(bat.columns)))
    bat.iColumns = make([]int, 0, imin(len(df.ints), len(bat.columns)))
    for i, col := range bat.columns {
      if _, ok := df.floats[col]; ok {
        bat.fColumns = append(bat.fColumns, i)
      } else if _, ok := df.bools[col]; ok {
        bat.bColumns = append(bat.bColumns, i)
      } else if _, ok := df.ints[col]; ok {
        bat.iColumns = append(bat.iColumns, i)
      } else if _, ok := df.objects[col]; ok {
        panic(fmt.Sprintf("%s cannot be converted to floats", col))
      } else {
        panic(fmt.Sprintf("column %s does not exist", col))
      }
    }
  }
}
