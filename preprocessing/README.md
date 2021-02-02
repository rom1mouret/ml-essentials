### GoDoc

[v0.1.0](https://pkg.go.dev/github.com/rom1mouret/ml-essentials@v0.1.0/preprocessing)

### Imports

```go
import preproc "github.com/rom1mouret/ml-essentials/preprocessing"
```

### Available preprocessing components

- [FloatImputer](float_imputer.go)
- [Scaler](scaler.go)
- [HashEncoder](hash_encoder.go)
- [OneHotEncoder](one_hot.go)
- [AutoPreprocessor](auto_preprocessor.go), a processor that combines the 4 components above.

Preprocessors follow these design principles:

##### 1. They decide on their own which columns are used for training

For example, `FloatImputer` will use *every* float column.
If you want to train `FloatImputer` on a subset of float columns, use `ColumnView` as follows:

```go
imputer := preproc.NewFloatImputer(preproc.FloatImputerOptions{Policy: Mean})
imputer.Fit(df.ColumnView("height", "age"))
```

##### 2. They can be run on dataframes with extra columns

Once the imputer is trained on a subset of columns like "height" and "age", it does not matter what other columns come along when performing the transformation:

```go
imputer.Fit(df.ColumnView("height", "age"))
imputer.TransformInplace(df.ColumnView("height", "age", "weight"))
```

In the example above, "weight" will be ignored.

##### 3. They implement the `Transform` interface

See [preprocessing/interfaces.go](interfaces.go)


##### 4. They are readily serializable in JSON

```go
// serialization
serialized, err = json.Marshal(preproc)
// deserialization
preproc = &preprocessing.AutoPreprocessor{}
json.Unmarshal([]byte(serialized), &preproc)
```

### Vectorization of categorical features

To one-hot strings, first run a `HashEncoder` to transform strings into integers. Then call `OneHotEncoder` to transform integer categories into boolean columns.
Later, we may implement an `OrdinalEncoder` as an alternative to `HashEncoder`, but the chance of hashing collision is extremely low on 64-bit systems, so I would recommend that you stick to `HashEncoder` on such systems.

To avoid any confusion, let me clarify that `HashEncoder` does *not* vectorize categories via [feature hashing](https://en.wikipedia.org/wiki/Feature_hashing). Vectorizing is the job of `OneHotEncoder` and `HashEncoder` does *not* project categories onto a lower-dimension space.
