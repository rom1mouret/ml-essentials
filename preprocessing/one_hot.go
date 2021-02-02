package preprocessing

import (
  "fmt"
  "github.com/rom1mouret/ml-essentials/utils"
  "github.com/rom1mouret/ml-essentials/dataframe"
)

type OneHotOptions struct {
  MissingPolicy   CategoryPolicy
  UnknownPolicy   CategoryPolicy
  KeepUsedColumns bool
}

// OneHotencoder is a json-serializable structure that transforms integer-typed
// categorical values into boolean columns.
// https://en.wikipedia.org/wiki/One-hot
type OneHotEncoder struct {
  CategoricalColumns []string
  NewColumns         []string
  Categories         map[string]map[int]string
  Fallback           map[string][]string
  Options            OneHotOptions
}

type CategoryPolicy int

const(
  // If -1 is found in the training data, then
  // - it will get its own category ;
  // - otherwise we impute with the most frequent category.
  SeparateCategoryIfSeen CategoryPolicy = iota

  // A separate category is created for unknown and missing.
  // The same category is used for both unknown and missing.
  SharedSeparateCategory

  // Unknown and missing get their own category.
  SeparateCategory

  // self-explanatory
  ImputeWithMostFrequent
  ReturnError
)

// NewOneHotEncoder allocates a new OneHotEncoder
func NewOneHotEncoder(options OneHotOptions) *OneHotEncoder {
  encoder := new(OneHotEncoder)
  encoder.Options = options
  encoder.Options.fix()

  return encoder
}

func (encoder *OneHotEncoder) workerFits(df *dataframe.DataFrame, q utils.StringQ) {
  opt := encoder.Options
  for col := q.Next(); len(col) > 0; col = q.Next() {
    defer q.Notify(utils.ProcessedJob{Key: col})
    // compute the frequency of each category
    access := df.Ints(col)
    freqs := make(map[int]int)
    for i := 0; i < access.Size(); i++ {
      freqs[access.Get(i)] += 1
    }
    // missing values seen in the training data
    _, missingSeen := freqs[-1]
    delete(freqs, -1)

    // find which category is the most frequent
    var mostCommon int
    if opt.MissingPolicy != ReturnError || opt.UnknownPolicy != ReturnError {
      maxFreq := -1
      for category, frequency := range freqs {
        if frequency > maxFreq {
          maxFreq = frequency
          mostCommon = category
        }
      }
    }
    // define the new columns
    categoryToNewCol := encoder.Categories[col]
    i := 0
    for category := range(freqs) {
      categoryToNewCol[category] = fmt.Sprintf("%s_onehot%d", col, i)
      i++
    }
    fallback := encoder.Fallback[col]
    if opt.MissingPolicy == SeparateCategoryIfSeen {
      if missingSeen {
        fallback[0] = fmt.Sprintf("%s_missing", col)
      } else {
        fallback[0] = categoryToNewCol[mostCommon]
      }
    } else if opt.MissingPolicy == SeparateCategory {
      fallback[0] = fmt.Sprintf("%s_missing", col)
    } else if opt.MissingPolicy == ImputeWithMostFrequent {
      fallback[0] = categoryToNewCol[mostCommon]
    }
    if opt.UnknownPolicy == SeparateCategory {
      fallback[1] = fmt.Sprintf("%s_unk", col)
    } else if opt.UnknownPolicy == ImputeWithMostFrequent {
      fallback[1] = categoryToNewCol[mostCommon]
    } else if opt.UnknownPolicy == SharedSeparateCategory {
      fallback[1] = fallback[0]
    }
  }
}

// Fit implements PreprocTraining and Transform interfaces.
func (encoder *OneHotEncoder) Fit(df *dataframe.DataFrame) error {
  categoricalCols := df.IntHeader().NameList()
  encoder.CategoricalColumns = categoricalCols

  // instantiate the maps
  encoder.Categories = make(map[string]map[int]string)
  encoder.Fallback = make(map[string][]string)

  // initialize map early on to avoid issues with concurrent access on writing
  for _, col := range categoricalCols {
    encoder.Categories[col] = make(map[int]string)
    encoder.Fallback[col] = make([]string, 2)
  }
  // fit each column separately
  q := df.CreateColumnQueue(categoricalCols)
  for i := 0; i < q.Workers; i++ {
    go encoder.workerFits(df, q)
  }
  q.Wait()

  // new columns
  newCols := make(map[string]bool)
  for _, categoryToNewCol := range encoder.Categories {
    for _, newCol := range categoryToNewCol {
      newCols[newCol] = true
    }
  }
  for _, fallbacks := range encoder.Fallback {
    fallback1 := fallbacks[0]
    fallback2 := fallbacks[1]
    if len(fallback1) > 0 {
      newCols[fallback1] = true
    } else if encoder.Options.MissingPolicy != ReturnError {
      panic("fallback value missing for missing categories")
    }
    if len(fallback2) > 0 {
       if fallback2 != fallback1 {
         newCols[fallback2] = true
       }
    } else if encoder.Options.UnknownPolicy != ReturnError {
      panic("fallback value missing for unknown categories")
    }
  }
  encoder.NewColumns = make([]string, len(newCols))
  i := 0
  for colName := range newCols {
    encoder.NewColumns[i] = colName
    i++
  }
  return nil
}

func (encoder *OneHotEncoder) workerTransforms(df *dataframe.DataFrame, q utils.StringQ) {
  var nilColumn dataframe.BoolAccess
  var unkColumn dataframe.BoolAccess
  opt := encoder.Options

  for catCol := q.Next(); len(catCol) > 0; catCol = q.Next() {
    notif := utils.ProcessedJob{Key: catCol}
    access := df.Ints(catCol)
    categoryToNewCol := encoder.Categories[catCol]
    if opt.MissingPolicy != ReturnError {
      nilColumn = df.Bools(encoder.Fallback[catCol][0])
    }
    if opt.UnknownPolicy != ReturnError {
      unkColumn = df.Bools(encoder.Fallback[catCol][1])
    }
    for i := 0; i < access.Size(); i++ {
      val := access.Get(i)
      if val == -1 {
        if opt.MissingPolicy == ReturnError {
          notif.Error = fmt.Errorf("missing-category option was not enabled during training")
          break
        }
        nilColumn.Set(i, true)
      } else {
        if newCol, ok := categoryToNewCol[val]; ok {
          df.Bools(newCol).Set(i, true)
        } else if opt.UnknownPolicy == ReturnError {
          notif.Error = fmt.Errorf("%d: unknown category", val)
          break
        } else {
          unkColumn.Set(i, true)
        }
      }
    }
    q.Notify(notif)
  }
}

// TransformeView implements PreprocTraining and Transform interfaces.
func (encoder *OneHotEncoder) TransformView(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
  result := df.View()
  if len(encoder.CategoricalColumns) == 0 {
    return result, nil
  }
  // allocate the new columns
  result.AllocBools(encoder.NewColumns...)

  // we'll run the transformation on multiple CPUs if possible
  q := df.CreateColumnQueue(encoder.CategoricalColumns)
  for i := 0; i < q.Workers; i++ {
    go encoder.workerTransforms(result, q)
  }
  for _, job := range q.Results() {
    if job.Error != nil {
      return nil, job.Error
    }
  }
  // remove the categorical columns that were transformed
  if !encoder.Options.KeepUsedColumns {
    result.Drop(encoder.CategoricalColumns...)
  }
  return result, nil
}

// TransformedColumns implements PreprocTraining and Transform interfaces.
func (encoder *OneHotEncoder) TransformedColumns() []string {
  return encoder.CategoricalColumns
}

func (opt* OneHotOptions) fix() {
  // make sure MissingPolicy == Shared never happen
  if opt.MissingPolicy == SharedSeparateCategory {
    if opt.UnknownPolicy != SharedSeparateCategory {
      // swap MissingPolicy and UnknownPolicy
      opt.MissingPolicy = opt.UnknownPolicy
      opt.UnknownPolicy = SharedSeparateCategory
    } else { // both policies are SharedSeparateCategory
      // choose a good default
      opt.MissingPolicy = SeparateCategoryIfSeen
    }
  }
  if opt.UnknownPolicy == SeparateCategoryIfSeen {
    // unknown categories are never seen, by definition
    opt.UnknownPolicy = SeparateCategory
  }
}

// func (encoder *OneHotEncoder) InverseTransformInplace() {
//   // TODO: lazily construct the inverse table
//   // protect the construction with a Mutex
// }
