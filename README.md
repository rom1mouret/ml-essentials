ml-essentials is a data frame library for Go in the same vein as [qota](https://github.com/go-gota/gota) and [qframe](https://github.com/tobgu/qframe).

It draws inspiration from [pandas](https://pandas.pydata.org/) and [numpy](https://numpy.org/).

Unlike [qota](https://github.com/go-gota/gota) and [qframe](https://github.com/tobgu/qframe),
ml-essentials doesn't cater for data scientists, e.g. with functions to load Excel files, SQL databases or functions to help with [EDA](https://en.wikipedia.org/wiki/Exploratory_data_analysis).
It is best suited for machine learning engineers who want to serve their models in a safe and predictable manner.
It is also smaller, with a focus on simplicity, stability and clarity.

I hope that ml-essentials is transparent enough for users to glance at their code and get a sense of what ml-essentials does under the hood and how much it is going to cost in CPU and RAM usage.
To illustrate my point, I am enumerating below all the view-returning functions.
Those features are only available through views, so the user has no choice but to spell out what his/her code should do. For instance, `shuffled := df.ShuffleView().Copy()` does exactly what it looks like.

```
(df *DataFrame) IndexView(indices []int) *DataFrame
(df *DataFrame) SliceView(from int, to int) *DataFrame
(df *DataFrame) MaskView(mask []bool) *DataFrame
(df *DataFrame) ColumnView(columns ...string) *DataFrame
(df *DataFrame) ShuffleView() *DataFrame
(df *DataFrame) SampleView(n int, replacement bool) *DataFrame
(df *DataFrame) SplitNView(n int) []*DataFrame
(df *DataFrame) SplitView(batchSize int) []*DataFrame
(df *DataFrame) SplitTrainTestViews(testingRatio float64) (*DataFrame, *DataFrame)
(df *DataFrame) SortedView(byColumn string) *DataFrame
(df *DataFrame) TopView(byColumn string, n int, ascending bool, sorted bool) *DataFrame
(df *DataFrame) ReverseView() *DataFrame
(df *DataFrame) HashStringsView(columns ...string) *DataFrame
(df *DataFrame) DetachedView(columns ...string) *DataFrame
(df *DataFrame) ResetIndexView() *DataFrame
(df *DataFrame) ShallowCopy() *DataFrame
(df *DataFrame) ColumnConcatView(dfs ...*DataFrame) (*DataFrame, error)
```

View-returning functions are guaranteed not to copy any large chunk of data.

### Documentation and examples

- [dataframe package](dataframe/)
- [preprocessing package](preprocessing/)
- [algorithms package](algorithms/)
- [A-to-Z example](examples/linreg.go)

### Benchmarks

dataset: [kddcup98](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html)

task: linear regression

|                             | ml-essentials CPU=1 | ml-essentials CPU=16 | python (pandas + pytorch) |
|-----------------------------|---------------------|----------------------|---------------------------|
| reading CSV                 | 18.3                | 3.3                  | 4.3                       |
| shuffling and splitting     | 0.003               | 0.003                | 0.4                       |
| preprocessing fit_transform | 2.4                 | 0.8                  | 2.2                       |
| linreg training (1 epoch)   | 6.9                 | 6.9                  | 3.4                       |
| preprocessor on test data   | 1                   | 0.5                  | 0.77                      |
| writing predictions         | 33                  | 4.7                  | 426                       |
| reading written rows        | 170                 | 71                   | 410                       |

The reason it takes so long to read/write predictions is because one-hot encoding creates over 20,000 columns.

Reproduction

```bash
cd examples
go run linreg.go -momentum=0.2 -epochs=1 -testratio=0.33 -batchsize 256 cup98LRN.txt TARGET_B CONTROLN
python3 linreg.py -momentum=0.2 -epochs=1 -testratio=0.33 -batchsize 256 cup98LRN.txt TARGET_B CONTROLN
```

### Design choices

##### Native types

Here are the benchmarks that have motivated my decision to use 3 native types alongside `interface{}`.
Those benchmarks measure the time to copy a slice at specific indices (from a slice of indices).

| type           | speed      | storage choice |  missing value |
|----------------|------------|----------------|----------------|
|`[]interface{}` | 4.51 ns/op | `[]interface{}`| nil            |
|`[]string`      | 4.26 ns/op | `[]interface{}`| nil            |
|`[]float64`     | 1.97 ns/op | `[]float64`    | NaN            |
|`[]int`         | 1.80 ns/op | `[]int`        | -1             |
|`[]bool`        | 1.38 ns/op | `[]bool`       | not applicable |


Float64 were chosen over float32 for the sake of compatibility with [gonum](https://github.com/gonum).


##### `interface{}` type for all the columns

Storing all the data slices as `interface{}` is sound.
For one thing, this requires only one `map[string]interface{}`.
By contrast, ml-essentials allocates 5 `map[string]T`, even when empty.
Also, some functions get to be very succinct, for instance
`rename` can move the data from one column to another without ever knowing what
type the data is of.

Ultimately, it was decided not to use `interface{}` for everything. Most functions
do rely on knowing the precise type and casting the values anyway. The first version
used `interface{}` everywhere and lots of type assertion errors popped up. Although
they were easy to fix, the new implementation brings more peace of mind.

### Roadmap


- functions to store/retrieve gonum's blas vectors in the df.objects map
- functions to store/retrieve/sort datetime objects in the df.objects map
- functions to create masks, e.g. mask := df.Test("age").Lower(15).Mask()
- smarter ColumnSmartConcat function
- ordinal encoder as an alternative to Hash Encoder
- more methods to RawData, like some sort of concat
- optimization of TopView
- more options to CSV reader and writer, such as BOM parsing
- inverse transform for OneHot
- `RepeatView(n int, bool interleaved)`
- more evaluation metrics, such as cross entropy
- reading/writing data in JSON
- release as a Go module

### External Contributions

ml-essentials is not affiliated with any organization.
Contributions are welcome.
