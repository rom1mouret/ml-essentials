This README covers the basics.

### GoDoc

[v0.1.0](https://pkg.go.dev/github.com/rom1mouret/ml-essentials@v0.1.0/dataframe)

### Imports

```go
import "github.com/rom1mouret/ml-essentials/dataframe"
```

### DataFrame construction

DataFrames accept 4 types of columns.

|type       |missing value  |comment|
|-----------|---------------|-------|
|float64    | NaN           |       |
|int        | -1            | meant to store categorical values |
|bool       | not supported | |
|interface{}| nil           | called "object" columns|

Strings are stored in the `interface{}` columns.
ml-essentials distinguishes between regular object columns and string columns by keeping around the names of the string columns.
Some functions are specialized for string columns, e.g. `Encode(newEncoding encoding.Encoding)`.

Storing categorical values is the preferred use of integer columns.
That said, you are free to use them to store any kind of integers, including negative integers.
Negative integers won't be treated as missing values unless you run [IntImputer](../preprocessing/README.md).

##### Construction with a DataBuilder

```go
builder := dataframe.DataBuilder{RawData: dataframe.NewRawData()}
builder.AddFloats("height", 170, 180, 165)
builder.AddStrings("name", "Karen", "John", "Sophie")
df := builder.ToDataFrame()
df.PrintSummary().PrintHead(-1, "%.3f")
```

##### Construction from a CSV file

```go
spec := dataframe.CSVReadingSpec{
  MaxCPU: -1,
  MissingValues: []string{"", " ", "NA","-"},
  IntAsFloat: true,
  BoolAsFloat: false,
  BinaryAsFloat: true,
}
rawdata, err := dataframe.FromCSVFile("/path/to/csvfile.csv", spec)
```

or
```go
rawdata, err := dataframe.FromCSVFilePattern("/path/to/csvdir/*.csv", spec)
```

### Column names

You can manipulate column names via the ColumnHeader structure.

```go
h := df.FloatHeader().And(df.IntHeader()).Except("target", "id").NameList()
```

### Iterate over a dataframe

##### Option 1: ColumnAccess

```go
height := df.Floats("height")
for i := 0; i < height.Size(); i++ {
  height.Set(i, height.Get(i) / 2)
}
```

##### Option 2: Gonum Batching

```go
batching := dataframe.NewDense64Batching([]string{"age", "height", "gender"})
for _, batch := range df.SplitView(params.BatchSize) {
  // get a gonum matrix with columns age, height and gender (in that order)
  rows := batching.DenseMatrix(batch)
}
```

##### Option 3: Row Iterator

```go
iterator := NewFloat32Iterator(df, []string{"age", "height", "gender"})
for row, rowIdx, _ := iterator.NextRow(); row != nil; row, rowIdx, _ = iterator.NextRow() {
  // row is a float32 slice
}
```

### Views

Views are dataframes that share data with other dataframes.
There is no `View` type and `DataFrame` type. Both are of type `DataFrame`.
Quick example:
```go
view := df.ShuffleView()
view.OverwriteInts("level", []int{4, 1, 2, 1})
```
Here `view` shares its data with `df`. This is useful in two ways.
First, `ShuffleView` doesn't copy the data, thus it is fast and memory-efficient.
Second, it allows you to overwrite `df`'s data from anywhere in your program.
The "side effects" section explains why this is an advantage when it comes to handling indexed data.

If you want to avoid such side effects, you can detach the view from its parent dataframe.
```go
view := df.ShuffleView().DetachedView("level")
view.OverwriteInts("level", []int{4, 1, 2, 1})
```
Now, `OverwriteInts` does not alter `df` because `view` has its own `level` data.
Other columns of `df` remain shared.

ml-essentials provides a variety of functions to manage data copies at a fine-grained level.

```javascript
View < TransferRawDataFrom < ShallowCopy < Unshare < DetachedView < Copy
```

On one side of the spectrum, `View` only copies pointers.
On the opposite side, `Copy` copies almost everything.
`View`, `DetachedView` and `Copy` cover 99% of the cases.

`View` is handy if you want to execute an in-place operation without altering the original dataframe, as in this example:
```go
view := df.View()
view.Rename("level", "degree")
```
Now, `view` and `df` still share their data, but their columns are named differently.

##### Side effects

Side effects are normally considered anti-patterns but they do facilitate manipulating indexed data.
For instance, consider this scenario:

1. at the top level, the data is separated into "features" and "metadata". Example of metadata: unique identifier, timestamps.
2. the model makes predictions from the features and predictions with low confidence are thrown away.
3. back to the top level, we combine "metadata" columns with predictions using the indices of high-confidence rows.

Step 3 is error-prone. With ml-essentials, the idiomatic way is to avoid separating "features" and "metadata" in the first place.
Instead, we would rely on views to enforce that the metadata always aligns with the features and predicted values.

Among the way [Pandas](https://pandas.pydata.org/) can solve this problem, it can combine "features" and "metadata" in an index-aware fashion, but
this makes `pandas.concat` error-prone in other scenarios, like when it fills dataframes with `NaN` where indices don't align, that is if
`ignore_index` is left to its default value.

### Filtering, masking and indexing

Unlike [Pandas](https://pandas.pydata.org) and [Numpy](https://numpy.org),
there is no syntactic sugar to create masks and index arrays. Sugar aside, this section will look familiar to Pandas and Numpy users.

If you want to filter rows where "age" is over 18, you can do so with `MaskView`:
```go
ages := df.Floats("age")
mask := df.EmptyMask()
for i := 0; i < ages.Size(); i++ {
  mask[i] = ages.Get(i) >= 18
}
view := df.MaskView(mask)
```

Getting a mask from `EmptyMask()` is advantageous because it recycles `[]bool` slices across dataframes, but it is not mandatory.

Equivalent filtering with `IndexView`:
```go
ages := df.Floats("age")
indices := make([]int, 0, ages.Size())
for i := 0; i < ages.Size(); i++ {
  if ages.Get(i) >= 18 {
    indices = append(indices, i)
  }
}
view := df.IndexView(indices)
```

In the future, we may add syntactic sugar for common scenarios, e.g. `Condition("age").Higher(18)`.

### Write in a dataframe

You can use the `Set` function as shown above.
Alternatively, you might find it more convenient to write an entire column in one line of code:

```go
df.OverwriteFloats64("height", []float64{170, 180, 165})
```

This is almost the same as:
```go
height := df.Floats("height")
height.Set(0, 170)
height.Set(1, 180)
height.Set(2, 165)
```

The only difference is that `OverwriteFloats64` will create a new column if it doesn't already exist.

### Complete example

This is an example taken from [linear_regression.go](../algorithms/linear_regression.go)

```go
import (
  "gonum.org/v1/gonum/mat"
  "github.com/rom1mouret/ml-essentials/dataframe"
)

func Predict(df *dataframe.DataFrame, batchSize int, resultColumn string) *dataframe.DataFrame {
  df = df.ResetIndexView() // makes batching.DenseMatrix faster

  // pre-allocation
  weights := mat.NewVecDense(len(reg.Weights), reg.Weights)
  pred := make([]float64, df.NumRows())

  // prediction
  batching := dataframe.NewDense64Batching(reg.Features)
  for i, batch := range df.SplitView(batchSize) {
    rows := batching.DenseMatrix(batch)
    offset := i * batchSize
    yData := pred[offset:offset+batch.NumRows()]
    yVec := mat.NewVecDense(len(yData), yData)
    yVec.MulVec(rows, weights)
  }

  // write the result in the output dataframe
  result := df.View()
  result.OverwriteFloats64("_target", pred)
  reg.TargetScaler.InverseTransformInplace(result)
  result.Rename("_target", resultColumn)

  return result
}
```
