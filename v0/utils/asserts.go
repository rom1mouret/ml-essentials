package utils

import (
  "testing"
  "reflect"
)

func AssertStringEquals(name string, actual string, expected string, t *testing.T) bool {
  if actual != expected {
    t.Errorf("%s = %s; expected: %s", name, actual, expected)
    return false
  }
  return true
}

func AssertStringNotEquals(name string, actual string, notExpected string, t *testing.T) bool {
  if actual == notExpected {
    t.Errorf("%s = %s", name, actual)
    return false
  }
  return true
}

func AssertIntEquals(name string, actual int, expected int, t *testing.T) bool {
  if actual != expected {
    t.Errorf("%s = %d; expected: %d", name, actual, expected)
    return false
  }
  return true
}

func AssertIntNotEquals(name string, actual int, notExpected int, t *testing.T) bool {
  if actual == notExpected {
    t.Errorf("%s = %d", name, actual)
    return false
  }
  return true
}

func AssertFloatEquals(name string, actual float64, expected float64, t *testing.T) bool {
  if actual < expected-0.01 || actual > expected+0.01 {
    t.Errorf("%s = %.4f; expected: %.4f", name, actual, expected)
    return false
  }
  return true
}

func AssertFalse(message string, value bool, t *testing.T) bool {
  if value {
    t.Errorf(message)
    return false
  }
  return true
}

func AssertTrue(message string, value bool, t *testing.T) bool {
  return AssertFalse(message, !value, t)
}

func AssertNoError(err error, t *testing.T) bool {
  if err != nil {
    t.Errorf(err.Error())
    return false
  }
  return true
}

func AssertStringSliceEquals(name string, actual []string, expected []string, orderMatters bool, t *testing.T) bool {
  err := false
  if orderMatters {
    if !reflect.DeepEqual(actual, expected) {
      err = true
    }
  } else {
      set1 := ToStringSet(actual)
      set2 := ToStringSet(expected)
      if !reflect.DeepEqual(set1, set2) {
        err = true
      }
  }
  if err {
    t.Errorf("%s = %s; expected: %s", name, actual, expected)
  }
  return !err
}

func AssertIntSliceEquals(name string, actual []int, expected []int, t *testing.T) bool {
  if !reflect.DeepEqual(actual, expected) {
    t.Errorf("%s = %v; expected: %v", name, actual, expected)
    return false
  }
  return true
}

func AssertIntSliceNotEquals(name string, actual []int, notExpected []int, t *testing.T) bool {
  if reflect.DeepEqual(actual, notExpected) {
    t.Errorf("%s = %v", name, actual)
    return false
  }
  return true
}

func AssertFloatSliceEquals(name string, actual []float64, expected []float64, t *testing.T) bool {
  if !reflect.DeepEqual(actual, expected) {
    if len(actual) != len(expected) {
      t.Errorf("%s = %v; expected: %v", name, actual, expected)
    } else {
        for i, v := range actual {
          AssertFloatEquals(name, v, expected[i], t)
        }
    }
    return false
  }
  return true
}

func AssertBoolSliceEquals(name string, actual []bool, expected []bool, t *testing.T) bool {
  if !reflect.DeepEqual(actual, expected) {
    t.Errorf("%s = %v; expected: %v", name, actual, expected)
    return false
  }
  return true
}
