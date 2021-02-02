package dataframe

// DataFrameInternals is a helper structure that lets you access the internals
// of a dataframe.
// Initialize the structure like that: DataFrameInternals{DF: your_df}
// Normally not needed, hence the lack of documentation.
type DataFrameInternals struct {
  DF *DataFrame
}

func (dfi DataFrameInternals) GetMask() []bool {
  return dfi.DF.mask
}

func (dfi DataFrameInternals) GetIndices() []int {
  return dfi.DF.indices
}

func (dfi DataFrameInternals) FloatData(columnName string) []float64 {
  return dfi.DF.floats[columnName]
}

func (dfi DataFrameInternals) IntData(columnName string) []int {
  return dfi.DF.ints[columnName]
}

func (dfi DataFrameInternals) BoolData(columnName string) []bool {
  return dfi.DF.bools[columnName]
}

func (dfi DataFrameInternals) ObjectData(columnName string) []interface{} {
  return dfi.DF.objects[columnName]
}
