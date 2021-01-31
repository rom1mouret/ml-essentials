package dataframe

import (
  "runtime"
  "encoding/csv"
  "math"
  "strconv"
  "io"
  "os"
  "path/filepath"
  "golang.org/x/text/encoding"
  "github.com/rom1mouret/ml-essentials/v0/utils"
)

type CSVReadingSpec struct {
  // This is to multi-thread the type conversions.
  // Zero and negative values mean ALL cpus on your machine.
  // The created RawData will also inherit from this value.
  MaxCPU int

  // Optional header if the CSV has no header.
  Header []string

  // Columns to exclude.
  Exclude []string

  // List of string literals that will be interpreted as missing values.
  MissingValues []string

  // Read integers and/or bool as floats.
  IntAsFloat    bool
  BoolAsFloat   bool  // 'true', 'false', '0' and '1' converted to 0.0 and 1.0
  BinaryAsFloat bool  // '0' and '1' converted to 0.0 and 1.0

  // How the CSV is encoded.
  // if not provided, it will ignore the encoding and fallback to UTF-8 if a
  // conversion is needed.
  // Note:
  // - the CSV is not decoded at reading.
  // - you can run nearly every function of ml-essentials without ever knowing
  //   the encoding
  Encoding encoding.Encoding
  // TODO: if not provided but BOM provided, read the BOM to detect the encoding

  // Options from https://golang.org/src/encoding/csv/reader.go
  Comma rune
  Comment rune
  LazyQuotes bool
	TrimLeadingSpace bool
}

func isBinary(records [][]string, col int) bool {
  for row := 0; row < len(records); row++ {
    val := records[row][col]
    if val != "0" && val != "1" {
      return false
    }
  }
  return true
}

func isBool(records [][]string, col int) bool {
  for row := 0; row < len(records); row++ {
    _, err := strconv.ParseBool(records[row][col])
    if err != nil {
      return false
    }
  }
  return true
}

func toBool(records [][]string, col int) []bool {
  values := make([]bool, len(records))
  for row := 0; row < len(records); row++ {
    v, _ := strconv.ParseBool(records[row][col])
    values[row] = v
  }
  return values
}

func isInt(records [][]string, col int, missing []bool) bool {
  for row := 0; row < len(records); row++ {
    if !missing[row] {
      _, err := strconv.ParseInt(records[row][col], 10, 64)
      if err != nil {
        return false
      }
    }
  }
  return true
}

func toInt(records [][]string, col int, missing []bool) []int {
  values := make([]int, len(records))
  for row := 0; row < len(records); row++ {
    if missing[row] {
      values[row] = -1
    } else {
      v, _ := strconv.ParseInt(records[row][col], 10, 64)
      values[row] = int(v)
    }
  }
  return values
}

func isFloat(records [][]string, col int, missing []bool) bool {
  for row := 0; row < len(records); row++ {
    if !missing[row] {
      _, err := strconv.ParseFloat(records[row][col], 64)
      if err != nil {
        return false
      }
    }
  }
  return true
}

func toFloat(records [][]string, col int, missing []bool) []float64 {
  values := make([]float64, len(records))
  for row := 0; row < len(records); row++ {
    if missing[row] {
      values[row] = math.NaN()
    } else {
      v, _ := strconv.ParseFloat(records[row][col], 64)
      values[row] = v
    }
  }
  return values
}

func toNativeTypes(data *RawData, records [][]string, header []string,
                   missingVals map[string]bool, spec CSVReadingSpec,
                   q utils.StringQ) {
  missing := make([]bool, len(records))
  for colName := q.Next(); len(colName) > 0; colName = q.Next() {
    col := utils.IndexOfString(colName, header)
    if !spec.BinaryAsFloat && isBinary(records, col) {
      data.bools[colName] = toBool(records, col)
    } else if !spec.BoolAsFloat && !spec.BinaryAsFloat && isBool(records, col) {
      data.bools[colName] = toBool(records, col)
    } else {
      if len(missingVals) > 0 {
        for i := range missing {
          missing[i] = missingVals[records[i][col]]
        }
      }
      if !spec.IntAsFloat && isInt(records, col, missing) {
        // detect missing values
        data.ints[colName] = toInt(records, col, missing)
      } else if isFloat(records, col, missing) {
        data.floats[colName] = toFloat(records, col, missing)
      } else {
        // fallback to strings
        values := make([]interface{}, len(records))
        for row := range values {
          if !missing[row] {
            values[row] = records[row][col]
          }
        }
        data.objects[colName] = values
        data.stringHeader.add(colName)
      }
    }
    q.Notify(utils.ProcessedJob{Key: colName})
  }
}

// FromCSV reads CSV data and returns a RawData structure with automatically
// inferred column types.
// It returns any error returned by golang's builtin CSV reader.
// With the default options, types are inferred this way:
// - If the column is 100% made of values that can be parsed as bools (0, 1,
// true, True, false, False or any other variant), it is stored as a bools.
// - Otherwise, if it is 100% made of integers or missing values, it is stored
// as an integer column. Integer missing values are replaced with -1.
// - Otherwise, if it is 100% made of floats or missing values, it is stored as
// a float column. Float missing values are replaced with NaN.
// - If none of the above match, the column is stored as a string column.
func FromCSV(r io.Reader, options CSVReadingSpec) (*RawData, error) {
  reader := csv.NewReader(r)
  reader.LazyQuotes = options.LazyQuotes
  reader.TrimLeadingSpace = options.TrimLeadingSpace
  if options.Comma != 0 {
    reader.Comma = options.Comma
  }
  if options.Comment != 0 {
    reader.Comment = options.Comment
  }
  // missing values
  missingVals := utils.ToStringSet(options.MissingValues)

  // TODO: accept unicode BOM here,
  // and use it to auto-detect the encoding

  // read the header
  var header []string
  if len(options.Header) == 0 {
    // header is in the file
    record, err := reader.Read()
    if err != nil {
      return nil, err
    }
    header = record
  } else {
    header = options.Header
  }
  // read all the data
  records, err := reader.ReadAll()
  if err != nil {
    return nil, err
  }
  // column queue for the worker pool
  colsToParse := Columns(header...).Except(options.Exclude...)
  tmp := RawData{} // only to get a col q
  tmp.SetMaxCPU(options.MaxCPU)
  q := tmp.CreateColumnQueue(colsToParse.NameList())

  // run the workers and store the results in protoframes
  protoframes := make([]*RawData, q.Workers)
  for i := range protoframes {
    emptyShell := NewRawData()
    go toNativeTypes(emptyShell, records, header, missingVals, options, q)
    protoframes[i] = emptyShell
  }
  for _, r := range q.Results() {
    if r.Error != nil {
      return nil, err
    }
  }
  // putting everything together
  result := MergeRawDataColumns(protoframes)
  result.textEncoding = options.Encoding
  result.dataUID = generateDataUID()
  result.resetStructureUID()
  result.SetMaxCPU(options.MaxCPU)

  return result, nil
}

// FromCSVFile reads a CSV file and returns a RawData structure with
// automatically inferred column types.
// It returns any error returned by golang's builtin CSV reader.
// It also returns an error if the file cannot be opened.
// For the type inference, refer to FromCSV's documentation.
func FromCSVFile(path string, options CSVReadingSpec) (*RawData, error) {
  f, err := os.Open(path)
  if err != nil {
    return nil, err
  }
  defer f.Close()
  return FromCSV(f, options)
}

// FromCSVFilePattern searches for file paths that matches the given glob
// pattern, reads them and returns a single RawData structure containing all the
// data packed in an unordered fashion.
// It returns any error returned by golang's builtin CSV reader.
// It also returns an error if any of the matching file can't be opened.
// If no file can be found, it returns (nil, nil).
// For the type inference, refer to FromCSV's documentation.
func FromCSVFilePattern(glob string, options CSVReadingSpec) (*RawData, error) {
  // list all the files we want to read
  filenames, err := filepath.Glob(glob)
  if err != nil {
    return nil, err
  }
  // keep only regular files
  paths := make([]string, 0, len(filenames))
  for _, filename := range filenames {
    fi, err := os.Stat(filename)
    if err != nil {
      return nil, err
    }
    if fi.Mode().IsRegular() {
      paths = append(paths, filename)
    }
  }
  if len(paths) == 0 {
    return nil, nil
  }
  // column-wise multi-threading
  if len(paths) == 1 {
    return FromCSVFile(paths[0], options)
  }
  // file-wise multi-threading
  cpu := options.MaxCPU
  if cpu <= 0 {
    cpu = runtime.NumCPU()
  }
  cpuPerFile := cpu / len(paths)
  if cpuPerFile == 0 {
    cpuPerFile = 1
  }
  options.MaxCPU = cpuPerFile
  maxWorkers := len(paths) / cpuPerFile
  fileQ := utils.CreateStringQueue(paths, maxWorkers)
  for i := 0; i < fileQ.Workers; i++ {
    go processFileQueue(&options, fileQ)
  }
  data := make([]*RawData, len(paths))
  for i, result := range fileQ.Results() {
    if result.Error != nil {
      return nil, result.Error
    }
    data[i] = result.Result.(*RawData)
  }
  // concatenate everything
  return MergeRawDataRows(data), nil
}

func processFileQueue(options* CSVReadingSpec, q utils.StringQ) {
  for path := q.Next(); len(path) > 0; path = q.Next() {
    data, err := FromCSVFile(path, *options)
    q.Notify(utils.ProcessedJob{Key: path, Error: err, Result: data})
  }
}
