package preprocessing

import (
  "math"
  "github.com/rom1mouret/ml-essentials/v0/utils"
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
)

type ImputingPolicy int
const(
  // replaces missing values with the average
  Mean  ImputingPolicy = iota
  // replaces missing values with zeros
  Zero
  // replaces missing values with ones
  One
  // TODO: add more options
)

// FloatImputer is a json-serializable structure that lets you replace missing
// values with "neutral" values computed at training time.
type FloatImputer struct {
  Fallback map[string]float64
  options FloatImputerOptions
}

type FloatImputerOptions struct {
  // specifies how to replace missing values
  Policy  ImputingPolicy
}

// NewFloatImputer allocates a new FloatImputer
func NewFloatImputer(options FloatImputerOptions) *FloatImputer {
  imputer := new(FloatImputer)
  imputer.options = options

  return imputer
}

// TransformedColumns implements PreprocTraining interface and InplaceTransform.
func (imputer *FloatImputer) TransformedColumns() []string {
  result := make([]string, len(imputer.Fallback))
  i := 0
  for key := range imputer.Fallback {
    result[i] = key
    i++
  }
  return result
}

func (imputer *FloatImputer) workerFits(df *dataframe.DataFrame, q utils.StringQ) {
  for col := q.Next(); len(col) > 0; col = q.Next() {
    access := df.Floats(col)
    sum := 0.0
    n := 0
    for j := 0; j < access.Size(); j++ {
      val := access.Get(j)
      if !math.IsNaN(val) {
        sum += val
        n++
      }
    }
    fallback := sum / float64(n)
    defer q.Notify(utils.ProcessedJob{Key: col, Result: &fallback})
  }
}

// Fit implements PreprocTraining interface and InplaceTransform.
func (imputer *FloatImputer) Fit(df *dataframe.DataFrame) error {
  columns := df.FloatHeader().NameList()
  imputer.Fallback = make(map[string]float64)
  if imputer.options.Policy == Zero {
    for _, col := range columns {
      imputer.Fallback[col] = 0
    }
  } else if imputer.options.Policy == One {
    for _, col := range columns {
      imputer.Fallback[col] = 1
    }
  } else { // Mean policy
    // run the calculations in separate threads
    q := df.CreateColumnQueue(columns)
    for i := 0; i < q.Workers; i++ {
      go imputer.workerFits(df, q)
    }
    for _, job := range q.Results() {
      imputer.Fallback[job.Key] = *job.Result.(*float64)
    }
  }
  return nil
}

// TransformInplace implements PreprocTraining interface and InplaceTransform.
func (imputer *FloatImputer) TransformInplace(df *dataframe.DataFrame) error {
  // TODO: multithread this?
  for col, fallback := range imputer.Fallback {
    access := df.Floats(col)
    for i := 0; i < access.Size(); i++ {
      if math.IsNaN(access.Get(i)) {
        access.Set(i, fallback)
      }
    }
  }
  return nil
}
