package dataframe

import (
  "sync"
  "sort"
  "encoding/csv"
  "fmt"
  "strconv"
  "io"
  "os"
)

type CSVWritingSpec struct {
  // missing values will be replaced with this string. Default: ""
  StringMissingMarker string
  // Value used by ToCSVDir when splitting the dataframe into multiple files.
  // If there are more rows than MinRowsPerFile, it will be split depending on
  // the MaxCPU attached to the dataframe.
  // Default: 512 rows
  MinRowsPerFile int
  // If maintaining the row order is not important, I encourage you to set this
  // value to False.
  MaintainOrder bool
  // Options from https://golang.org/src/encoding/csv/writer.go
  Comma   rune // Field delimiter (set to ',' by NewWriter)
  UseCRLF bool // True to use \r\n as the line terminator

  // TODO: BOM writing, maybe?
}

// To1CSV writes the dataframe in CSV format into the writer given as argument.
// It returns an error if the writer doesn't allow writing.
// It also forwards any error raised by golang's builtin CSV writer.
// To1CSV flushes the writer before returning.
// This function is not multi-threaded.
func (df *DataFrame) To1CSV(r io.Writer, options CSVWritingSpec) error {
  writer := csv.NewWriter(r)
  if options.Comma != 0 {
    writer.Comma = options.Comma
  }
  writer.UseCRLF = options.UseCRLF
  defer writer.Flush()
  // this minimize the cache-misses
  df = df.sortIfNeeded(options)

  // write the header (bool, int, float, strings)
  bCols := df.BoolHeader().NameList()
  iCols := df.IntHeader().NameList()
  fCols := df.FloatHeader().NameList()
  sCols := df.StringHeader().NameList()
  colNames := append(append(append(bCols, iCols...), fCols...), sCols...)
  err := writer.Write(colNames)
  if err != nil {
    return err
  }
  // pre-allocate some rows to amortize the cost of iterating through the maps
  numRows := df.NumRows()
  var batch [256][]string
  for i := 0; i < len(batch); i++ {
    batch[i] = make([]string, len(colNames))
  }
  for j := 0; j < numRows; j += len(batch) {
    end := j + len(batch)
    if end > numRows {  //TODO: use IndexSpliter instead?
      end = numRows
    }
    // we iterate over the rows in the same order as colNames
    col := 0
    for _, colName := range bCols {
      vals := df.bools[colName]
      for i, k := range df.indices[j:end] {
        batch[i][col] = strconv.FormatBool(vals[k])
      }
      col++
    }
    for _, colName := range iCols {
      vals := df.ints[colName]
      for i, k := range df.indices[j:end] {
        batch[i][col] = strconv.Itoa(vals[k])
      }
      col++
    }
    for _, colName := range fCols {
      vals := df.floats[colName]
      for i, k := range df.indices[j:end] {
        batch[i][col] = strconv.FormatFloat(vals[k], 'f', -1, 64)
      }
      col++
    }
    for _, colName := range sCols {
      vals := df.objects[colName]
      for i, k := range df.indices[j:end] {
        val := vals[k]
        if val == nil {
          batch[i][col] = options.StringMissingMarker
        } else {
          batch[i][col] = val.(string)
        }
      }
      col++
    }
    // write the batch
    err := writer.WriteAll(batch[:end - j])
    if err != nil {
      return err
    }
  }
  return nil
}

func (df *DataFrame) threadedTo1CSV(w io.Writer, err []error, opt CSVWritingSpec, wg *sync.WaitGroup) {
  defer wg.Done()
  err[0] = df.To1CSV(w, opt)
}

// To1CSV writes the dataframe in CSV format into the writers given as argument.
// The dataframe is split evenly between the writers and each writer is called
// separately within their dedicated go routine.
// The row order is not guaranteed.
// It returns an error if one of the writers doesn't allow writing.
// It also forwards any error raised by golang's builtin CSV writer.
func (df *DataFrame) ToCSVs(writers []io.Writer, options CSVWritingSpec) error {
  /// divide the input data
  dfs := df.sortIfNeeded(options).SplitNView(len(writers))
  options.MaintainOrder = true  // already dealt with by sortIfNeeded

  // write each data in a thread
  errors := make([]error, len(writers))  // TODO: use a channel instead
  var wg sync.WaitGroup
  wg.Add(len(writers))
  for i, w := range writers {
    go dfs[i].threadedTo1CSV(w, errors[i:i+1], options, &wg)
  }
  wg.Wait()
  // report errors
  for _, err := range errors {
    if err != nil {
      return err
    }
  }
  return nil
}

// ToCSVFiles writes the dataframe in CSV format in the given files.
// The dataframe is split evenly between the files and each file is written
// separately within their dedicated go routine.
// It returns an error if one of the files doesn't allow writing.
// It also forwards any error raised by golang's builtin CSV writer.
func (df *DataFrame) ToCSVFiles(options CSVWritingSpec, paths ...string) error {
  writers := make([]io.Writer, len(paths))
  for i, path := range paths {
    f, err := os.Create(path)
    defer f.Close()
    if err != nil {
      return err
    }
    writers[i] = f
  }
  return df.ToCSVs(writers, options)
}

// ToCSVDir writes the dataframe in CSV format to files with the chosen prefix.
// The prefix includes the directory.
// Example of prefix: "/tmp/output/result"
// This will write /tmp/output/result01.csv, /tmp/output/result02.csv etc.
// The dataframe is split evenly between the files and each file is written
// separately within their dedicated go routine.
// It returns an error if one of the files doesn't allow writing or if the
// output directory does not exist.
// It also forwards any error raised by golang's builtin CSV writer.
// Alongside the potential error, ToCSVDir returns the list of files written.
func (df *DataFrame) ToCSVDir(options CSVWritingSpec, prefix string) ([]string, error) {
  // number of files based on the max number of CPUs
  if options.MinRowsPerFile <= 0 {
    options.MinRowsPerFile = 512
  }
  numFiles := 1 + df.NumRows() / options.MinRowsPerFile
  if df.ActualMaxCPU() < numFiles {
    numFiles = df.ActualMaxCPU()
  }
  // construct paths
  suffix := "%d.csv"
  if numFiles >= 100 {
    suffix = "%03d.csv"
  } else if numFiles >= 10 {
    suffix = "%02d.csv"
  }
  paths := make([]string, numFiles)
  for i := range paths {
    paths[i] = prefix + fmt.Sprintf(suffix, i)
  }
  // write the CSV files
  return paths, df.ToCSVFiles(options, paths...)
}

func (df *DataFrame) sortIfNeeded(options CSVWritingSpec) *DataFrame {
  if df.indexViewed && !options.MaintainOrder {
    result := df.View()
    result.indices = make([]int, len(df.indices))
    sort.Ints(result.indices)
    return result
  }
  return df
}
