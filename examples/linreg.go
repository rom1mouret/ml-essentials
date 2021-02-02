package main

import (
  "flag"
  "log"
  "time"
  "fmt"
  "encoding/json"
  "github.com/rom1mouret/ml-essentials/dataframe"
  "github.com/rom1mouret/ml-essentials/preprocessing"
  "github.com/rom1mouret/ml-essentials/algorithms"
)

func main() {
  lr := flag.Float64("lr", 0.0001, "gradient descent learning rate")
  decay := flag.Float64("decay", 0.0, "weight decay")
  //clipping := flag.Float64("clipping", 0.0, "gradient clipping")
  Momentum := flag.Float64("momentum", 0.0, "Momentum")
  split := flag.Float64("testratio", 0.5, "ratio of testing rows")
  imputing := flag.Bool("imputing", true, "enable imputing")
  scaling := flag.Bool("scaling", true, "enable centering and scaling")
  lowmem := flag.Bool("lowmem", false, "low memory => avoids copies")
  epochs := flag.Int("epochs", 1, "number of training epochs")
  batchSize := flag.Int("batchsize", 64, "number of rows in one batch")
  maxCPU := flag.Int("cpu", -1, "maximum number of parallel routines")
  sep := flag.String("sep", ",", "character delimiter in the CSV file")
  debugging := flag.Bool("debug", false, "enable debugging (major slow down)")
  flag.Parse()
  rest := flag.Args()
  csvFile := rest[0]
  targetColumn := rest[1]
  exclude := rest[2:flag.NArg()]
  runeComma := rune([]byte(*sep)[0])

  // linear regressor
  modelOpt := algorithms.LinRegTrainParams{
    Epochs:      *epochs,
    LR:          *lr,
    BatchSize:   *batchSize,
    WeightDecay: *decay,
    Momentum:    *Momentum,
    LowMemory:   *lowmem,
    Verbose:     true,
  }
  fmt.Println("options:", modelOpt, "target:", targetColumn, "exclude:", exclude)
  model := algorithms.NewLinearRegressor()

  // read the data
  start := time.Now()
  spec := dataframe.CSVReadingSpec{
    MaxCPU: *maxCPU,
    MissingValues: []string{"", " ", "NA","-"},
    IntAsFloat: true,
    BoolAsFloat: false,
    BinaryAsFloat: true,
    Comma: runeComma,
    Exclude: exclude,
  }
  rawdata, err := dataframe.FromCSVFile(csvFile, spec)
  if err != nil {
    panic(err.Error())
  }
  df := rawdata.ToDataFrame().Debug(*debugging)
  log.Printf("reading %s took %s", csvFile, time.Since(start))
  df.PrintSummary()

  // split the dataset for cross-validation
  start = time.Now()
  trainSet, testSet := df.ShuffleView().SplitTrainTestViews(*split)
  log.Printf("shuffling+splitting the dataset took %s", time.Since(start))

  // train and run the preprocessor
  start = time.Now()
  preprocOpt := preprocessing.AutoPreprocOptions{
    Imputing: *imputing,
    Scaling: *scaling,
    Verbose: true,
    Exclude: []string{targetColumn},
  }
  preproc := preprocessing.NewAutoPreprocessor(preprocOpt)
  trainSet = preproc.FitTransform(trainSet)
  log.Printf("total preprocessing training took %s", time.Since(start))

  // main model
  model.Fit(trainSet, targetColumn, modelOpt)
  log.Printf("total training took %s", time.Since(start))

  //serialization/deserialization (for testing purpose only)
  serialized, err := json.Marshal(model)
  if err != nil {
    panic(err.Error())
  }
  model = &algorithms.LinearRegressor{}
  json.Unmarshal([]byte(serialized), &model)

  serialized, err = json.Marshal(preproc)
  if err != nil {
    panic(err.Error())
  }
  preproc = &preprocessing.AutoPreprocessor{}
  json.Unmarshal([]byte(serialized), &preproc)

  // run preprocessor on test set
  start = time.Now()
  testSet, err = preproc.TransformView(testSet)
  if err != nil {
    panic(err.Error())
  }
  log.Printf("total preprocessor for predicting took %s with cpu=%d",
             time.Since(start), testSet.ActualMaxCPU())

  // predicting
  start = time.Now()
  testSet, _ = model.Predict(testSet, "y_pred")
  log.Printf("predicting took %s", time.Since(start))

  // evaluation
  mae := algorithms.MAE(testSet.Floats(targetColumn), testSet.Floats("y_pred"))
  log.Printf("MAE: %f", mae)

  // some i/o for testing
  start = time.Now()
  testSet.ToCSVDir(dataframe.CSVWritingSpec{Comma: runeComma}, "outdir/file")
  log.Printf("total writing %d rows to directory %s", testSet.NumRows(), time.Since(start))
  start = time.Now()
  newdata, err := dataframe.FromCSVFilePattern("outdir/*.csv", spec)
  if err != nil {
    panic(err.Error())
  }
  if newdata == nil {
    panic("no data in outdir")
  }
  log.Printf("total reading %d rows from directory %s",
             newdata.NumAllocatedRows(), time.Since(start))
  newdata.ToDataFrame()
}
