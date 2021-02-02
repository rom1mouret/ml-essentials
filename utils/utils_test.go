package utils

import "testing"

func Test1SplitStringSlice(t *testing.T) {
  result := SplitStringSlice([]string{"a", "b", "c"}, 2)
  if AssertIntEquals("len(result)", len(result), 2, t) {
    AssertStringSliceEquals("result[0]", result[0], []string{"a", "b"}, true, t)
    AssertStringSliceEquals("result[1]", result[1], []string{"c"}, true, t)
  }
}

func Test2SplitStringSlice(t *testing.T) {
  result := SplitStringSlice([]string{"a", "b", "c", "d"}, 2)
  if AssertIntEquals("len(result)", len(result), 2, t) {
    AssertStringSliceEquals("result[0]", result[0], []string{"a", "b"}, true, t)
    AssertStringSliceEquals("result[1]", result[1], []string{"c", "d"}, true, t)
  }
}

func Test3SplitStringSlice(t *testing.T) {
  result := SplitStringSlice([]string{"a", "b"}, 3)
  if AssertIntEquals("len(result)", len(result), 2, t) {
    AssertStringSliceEquals("result[0]", result[0], []string{"a"}, true, t)
    AssertStringSliceEquals("result[1]", result[1], []string{"b"}, true, t)
  }
}

func Test4SplitStringSlice(t *testing.T) {
  result := SplitStringSlice([]string{"a", "b", "c"}, 3)
  if AssertIntEquals("len(result)", len(result), 3, t) {
    AssertStringSliceEquals("result[0]", result[0], []string{"a"}, true, t)
    AssertStringSliceEquals("result[1]", result[1], []string{"b"}, true, t)
    AssertStringSliceEquals("result[2]", result[2], []string{"c"}, true, t)
  }
}
