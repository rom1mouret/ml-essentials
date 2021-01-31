// This package is intended to be used by other ml-essentials' packages.
// There is nothing of value here for as far as end users are concerned.
package utils

import (
  "gonum.org/v1/gonum/floats"
)

func MakeRange(from int, to int, interval int) []int {
  result := make([]int, (to - from) / interval)
  for n, j := from, 0; n < to; n, j = n + interval, j+1 {
    result[j] = n
  }
  return result
}

func FloatArgSort(values []float64, readonly bool) []int {
  // unlike gonum, this doesn't change the input vector
  // TODO: find a better way than using gonum
  if readonly {
    cpy := make([]float64, len(values))
    copy(cpy, values)
    values = cpy
  }
  inds := make([]int, len(values))
  floats.Argsort(values, inds)

  return inds
}

func SplitStringSlice(strings []string, nGroups int) [][]string {
  if len(strings) == 0 || nGroups == 0 {
    return nil
  }
  indexer := CreateGroupIndexer(len(strings), nGroups)
  result := make([][]string, indexer.NGroups)
  for indexer.HasNext() {
    i, left, right := indexer.Next()
    result[i] = strings[left:right]
  }
  return result
}

func HashIntSlice(indices []int) int {
  result := len(indices)
  for _, v := range indices {
    result = 7 * result + v  // yes, this will overflow
  }
  return result
}

func ToStringSet(strings []string) map[string]bool {
  result := make(map[string]bool)
  for _, s := range strings {
    result[s] = true
  }
  return result
}

func StringSetKeys(set map[string]bool) []string {
  result := make([]string, len(set))
  i := 0
  for key := range set {
    result[i] = key
    i++
  }
  return result
}

func IndexOfString(element string, array []string) int {
   for k, v := range array {
       if element == v {
           return k
       }
   }
   return -1
}
