package preprocessing

import (
  "time"
  "log"
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
)

type AutoPreprocOptions struct {
  // Whether the data should be imputed or not.
  Imputing  bool
  // Whether the data should be scaled and centered with std and mean.
  Scaling   bool
  // Whether to print out some information such as training time.
  Verbose   bool
  // Columns to exclude from the transformations (e.g. the target of a model).
  Exclude   []string
}

// AutoPreprocessor is a structure that does most of the basic preprocessing for
// you. It encapsulates a string-to-int encoder, a category encoder, an imputer
// and a scaler (including centering).
// It is json-serializable.
type AutoPreprocessor struct {
  StringToInt *HashEncoder
  IntToBool   *OneHotEncoder
  Imputer     *FloatImputer
  Scaler      *Scaler
  options     AutoPreprocOptions
}

// NewAutoPreprocessor allocates a new untrained AutoPreprocessor
func NewAutoPreprocessor(opt AutoPreprocOptions) *AutoPreprocessor {
  result := new(AutoPreprocessor)
  result.options = opt
  return result
}

// TransformedColumns implements PreprocTraining and Transform interfaces.
func (preproc *AutoPreprocessor) TransformedColumns() []string {
  cols := preproc.Imputer.TransformedColumns()  // can be safely changed
  cols = append(cols, preproc.StringToInt.TransformedColumns()...)
  return cols
}

// Fit trains the AutoPreprocessor on the given dataframe and returns an error
// if an error occurred.
func (preproc *AutoPreprocessor) Fit(df *dataframe.DataFrame) error {
  _, err := preproc.fit(df, false)
  return err
}

// Fit trains the AutoPreprocessor on the given dataframe, transforms the
// dataframe, and returns the transformed dataframe.
// It is functionally equivalent to Fit()+TransformView() but is a bit faster.
// It will panic if an error occurred during the training or the transformation.
// The preprocessing options used by AutoPreprocessor should not give rise to
// any error though.
func (preproc *AutoPreprocessor) FitTransform(df *dataframe.DataFrame) *dataframe.DataFrame {
  result, err := preproc.fit(df, true)
  if err != nil {
    panic(err.Error())
  }
  return result
}

func (preproc *AutoPreprocessor) fit(df *dataframe.DataFrame, transform bool) (*dataframe.DataFrame, error) {
  opt := preproc.options
  floatCols := df.FloatHeader().Except(opt.Exclude...).NameList()
  if opt.Imputing || opt.Scaling {
    start := time.Now()
    floatsOnly := df.ColumnView(floatCols...)
    if opt.Imputing {
      preproc.Imputer = NewFloatImputer(FloatImputerOptions{})
      preproc.Imputer.Fit(floatsOnly)
    }
    if opt.Scaling {
      preproc.Scaler = NewScaler(ScalerOptions{Centering: true, Scaling: true})
      preproc.Scaler.Fit(floatsOnly)
    }
    if opt.Verbose {
      log.Printf("training Imputer+Scaler took %s with cpu=%d",
                 time.Since(start), floatsOnly.ActualMaxCPU())
    }
  }
  // category encoding
  var categorical []string
  if df.StringHeader().Num() > 0 {
    start := time.Now()
    preproc.StringToInt = NewHashEncoder(HashEncoderOptions{})
    preproc.StringToInt.Fit(df)
    view, err := preproc.StringToInt.TransformView(df)
    df = view
    if err != nil {
      return nil, err
    }
    categorical = preproc.StringToInt.TransformedColumns()
    if opt.Verbose {
      log.Printf("running Hash Encoder took %s with cpu=%d", time.Since(start), df.ActualMaxCPU())
    }
  } else {
    categorical = df.IntHeader().NameList()
  }
  if len(categorical) > 0 {
    start := time.Now()
    topt := OneHotOptions{
        MissingPolicy: SeparateCategoryIfSeen,
        UnknownPolicy: SharedSeparateCategory,
    }
    preproc.IntToBool = NewOneHotEncoder(topt)
    preproc.IntToBool.Fit(df.ColumnView(categorical...))
    if opt.Verbose {
      log.Printf("training One-Hot Encoder took %s with cpu=%d",
                 time.Since(start), df.ActualMaxCPU())
    }
  }
  if !transform{
    return nil, nil
  }
  // transform
  if preproc.IntToBool != nil {
    view, err := preproc.IntToBool.TransformView(df)
    df = view
    if err != nil {
      return nil, err
    }
  } else if !opt.Imputing && !opt.Scaling {
    return df, nil
  }
  start := time.Now()
  df = df.View()
  df.Unshare(floatCols...) // unshare() make it safe for in-place modifications
  preproc.Imputer.TransformInplace(df)
  preproc.Scaler.TransformInplace(df)
  if opt.Verbose {
    log.Printf("running Imputer+Scaler took %s with cpu=%d",
               time.Since(start), df.ActualMaxCPU())
  }
  return df, nil
}

// TransformView applies AutoPreprocessor's preprocessing components to the
// dataframe and returns a preprocessed dataframe.
// Because it returns a view, columns that don't require preprocessing will
// share their data with the columns of the original dataframe.
func (preproc *AutoPreprocessor) TransformView(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
  // prepare transformations in place
  if preproc.Imputer != nil {
    df = df.DetachedView(preproc.Imputer.TransformedColumns()...)
  } else if preproc.Scaler != nil {
    df = df.DetachedView(preproc.Scaler.TransformedColumns()...)
  }
  // transformations
  if preproc.Imputer != nil {
    preproc.Imputer.TransformInplace(df)
  }
  if preproc.Scaler != nil {
    preproc.Scaler.TransformInplace(df)
  }
  if preproc.StringToInt != nil {
    view, err := preproc.StringToInt.TransformView(df)
    df = view
    if err != nil {
      return nil, err
    }
  }
  if preproc.IntToBool != nil {
    view, err := preproc.IntToBool.TransformView(df)
    df = view
    if err != nil {
      return nil, err
    }
  }
  return df, nil
}
