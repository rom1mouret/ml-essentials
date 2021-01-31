package preprocessing

import (
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
)

type PreprocTraining interface {
  // TransformedColumns returns the columns used by the preprocessor.
  // Those columns are mandatory inputs. Other columns will be forwarded to the
  // output.
  // In the general case, the returned list shouldn't be changed.
  TransformedColumns() []string

  // Fit trains the preprocessor.
  // Training options are normally given to the preprocessor upon its
  // instantiation.
  // The preprocessor typically chooses the columns that are relevant to the
  // transformation. If you want the transformation to apply on a subset of
  // columns, you can write it like that:
  // preproc.Fit(df.ColumnView("this-column", "and-that-column"))
  // This won't stop you from running the preprocessor on a wider dataframe:
  // preproc.TransformInplace(df) // here 'df' has more than 2 columns.
  Fit(df *dataframe.DataFrame) error
}

type Transform interface {
  PreprocTraining
  // TransformView applies the transformation to the columns returned by
  // TransformedColumns() without changing the input dataframe. Instead, it
  // returns a shallow copy of the input dataframe wherein all the transformed
  // columns are newly allocated.
  // Columns that are not transformed are shallow copied to the output
  // dataframe, i.e. they share their data with the input dataframe.
  // It returns an error if an error occurred, e.g. a categorical feature of df
  // contains an unknown category.
  // TransformView is functionally equivalent to:
  // result := df.View()
  // result.TransformInplace()
  TransformView(df *dataframe.DataFrame) (*dataframe.DataFrame, error)
}

type InplaceTransform interface {
  PreprocTraining
  // TransformInplace applies the transformation in-place. That is, no new
  // dataframe is allocated.
  // It returns an error if an error occurred, e.g. a categorical feature of df
  // contains an unknown category.
  // You can emulate TransformView as follows:
  // view := df.View()
  // view.TransformInplace()
  TransformInplace(df *dataframe.DataFrame) error
}

type InverseInplaceTransform interface {
  InplaceTransform
  // InverseTransformInplace performs the inverse of whatever TransformInplace
  // does. For example, if a TransformInplace divides a column by 2,
  // InverseTransformInplace multiplies this column by 2.
  // Everything else follows TransformInplace's specifications.
  InverseTransformInplace(df *dataframe.DataFrame) error
}
