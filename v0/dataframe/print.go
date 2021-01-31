package dataframe

import (
   "fmt"
   "sort"
   "log"
   "reflect"
)

// PrintSummary prints information about the content of the dataframe, such as
// the name of the columns and the number of rows.
// It doesn't print the data.
// Everything is printed on stdout. Nothing on stderr.
// PrintSummary returns the dataframe itself so you can write
// df.PrintSummary().PrintHead(n, "") or df.PrintHead(n, "").PrintSummary()
func (df *DataFrame) PrintSummary() *DataFrame {
  if len(df.floats) > 0 {
    cols := df.FloatHeader().NameList()
    sort.Strings(cols)
    fmt.Printf("float   %s\n", cols)
  }
  if len(df.ints) > 0 {
    cols := df.IntHeader().NameList()
    sort.Strings(cols)
    fmt.Printf("int     %s\n", cols)
  }
  if len(df.bools) > 0 {
    cols := df.BoolHeader().NameList()
    sort.Strings(cols)
    fmt.Printf("bool    %s\n", cols)
  }
  if len(df.objects) > 0 {
    cols := df.ObjectHeader().ExceptHeader(df.StringHeader()).NameList()
    sort.Strings(cols)
    fmt.Printf("object  %s\n", cols)
  }
  strings := df.StringHeader().NameList()
  if len(strings) > 0 {
    sort.Strings(strings)
    fmt.Printf("string  %s\n", strings)
  }
  if df.shared.Num() > 0 {
    list := df.shared.NameList()
    sort.Strings(list)
    fmt.Printf("viewed  %s\n", list)
  }
  if df.sharedMaps {
    fmt.Printf("shared  YES (structure ID: %d)\n", df.structureUID)
  }
  fmt.Printf("rows    %d\n", len(df.indices))
  fmt.Printf("max CPU %d\n", df.ActualMaxCPU())
  if df.indexViewed {
    fmt.Printf("row-wise view: %d / %d rows\n", len(df.indices), len(df.mask))
  }
  return df
}

// PrintHead prints the first n rows of the dataframe.
// If n is negative or if n is greater than the number of rows, it will print
// all the rows.
// floatFormat describes how you want floats to be printed, e.g. %.4f
// floatFormat defaults to %.3f
// Everything is printed on stdout. Nothing on stderr.
// PrintHead returns the dataframe itself so you can write
// df.PrintSummary().PrintHead(n, "") or df.PrintHead(n, "").PrintSummary()
func (df *DataFrame) PrintHead(n int, floatFormat string) *DataFrame {
  if n < 0 || n > len(df.indices) {
    n = df.NumRows()
  }
  if len(floatFormat) == 0 {
    floatFormat = " %.3f"
  } else {
    floatFormat = " " + floatFormat
  }
  indices := df.indices[:n]
  if len(df.floats) > 0 {
    cols := df.FloatHeader().NameList()
    sort.Strings(cols)
    for _, col := range cols {
      vals := df.floats[col]
      fmt.Printf("%s:", col)
      for _, j := range indices {
        fmt.Printf(floatFormat, vals[j])
      }
      fmt.Println("")
    }
  }
  if len(df.ints) > 0 {
    cols := df.IntHeader().NameList()
    sort.Strings(cols)
    for _, col := range cols {
      vals := df.ints[col]
      fmt.Printf("%s:", col)
      for _, j := range indices {
        fmt.Printf(" %d", vals[j])
      }
      fmt.Println("")
    }
  }
  if len(df.bools) > 0 {
    cols := df.BoolHeader().NameList()
    sort.Strings(cols)
    for _, col := range cols {
      vals := df.bools[col]
      fmt.Printf("%s:", col)
      for _, j := range indices {
        printBool(vals[j])
      }
      fmt.Println("")
    }
  }
  if len(df.objects) > 0 {
    cols := df.ObjectHeader().NameList()
    sort.Strings(cols)
    for _, col := range cols {
      vals := df.objects[col]
      fmt.Printf("%s:", col)
      isString := df.stringHeader.contains(col)
      for _, j := range indices {
        df.printObject(vals[j], isString)
      }
      fmt.Println("")
    }
  }
  return df
}

// PrintRecords does the same as PrintHead but prints one line for each row.
// shorthands maps column names to shorter column names in order to avoid
// cluttering the output. You can leave it empty, nil, or call GoodShortNames()
// to get optimally small truncated names.
func (df *DataFrame) PrintRecords(n int, floatFormat string, shorthands map[string]string) *DataFrame {
  if n < 0 || n > len(df.indices) {
    n = df.NumRows()
  }
  if len(floatFormat) == 0 {
    floatFormat = " %.3f"
  } else {
    floatFormat = " " + floatFormat
  }
  order := df.Header().NameList()
  sort.Strings(order)
  indexFormat := fmt.Sprintf("[%s:%%d]", intFormat(n-1))
  for i, j := range df.indices {
    if i >= n {
      break
    }
    fmt.Printf(indexFormat, i, j)
    for _, colName := range order {
      // print the column name
      fmt.Printf(" ")
      if shorthand, defined := shorthands[colName]; defined {
        fmt.Printf(shorthand)
      } else {
        fmt.Printf(colName)
      }
      fmt.Printf(":")
      // print the value
      if val, ok := df.floats[colName]; ok {
        fmt.Printf(floatFormat, val[j])
      } else if val, ok := df.objects[colName]; ok {
        df.printObject(val[j], df.stringHeader.contains(colName))
      } else if val, ok := df.ints[colName]; ok {
        fmt.Printf(" %d", val[j])
      } else if val, ok := df.bools[colName]; ok {
        printBool(val[j])
      } else {
        fmt.Printf(" @ERROR@")
      }
    }
    fmt.Println("")
  }
  return df
}

// GoodShortNames returns short versions of column names for PrintRecords().
// minLength is the minimum length of shortened names.
// If minLength is zero or negative, it will default to minLength=3.
// This function is not deterministic.
func (df *DataFrame) GoodShortNames(minLength int) map[string]string {
  if minLength <= 0 {
    minLength = 3
  }
  invResult := make(map[string]string)
  for col := range df.Header().NameSet() {
    for length := minLength; length <= len(col); length++ {
      candidate := col[:length]
      if _, existing := invResult[candidate]; !existing {
        invResult[candidate] = col
        break
      }
    }
  }
  result := make(map[string]string)
  for shorthand, colName := range invResult {
    result[colName] = shorthand
  }
  return result
}

func (df *DataFrame) printObject(v interface{}, isString bool) {
  if v == nil {
    fmt.Printf(" <missing>")
  } else if isString && df.textEncoding == nil {
    fmt.Printf(" '%s'", v.(string))
  } else if isString {
    fmt.Printf(" <NOT-UTF8>")
  } else if reflect.ValueOf(v).Kind() == reflect.Ptr {
    fmt.Printf(" %v", v) // address of the object
  } else {
    fmt.Printf(" %v", &v) // address of the object
  }
}

func printBool(val bool) {
  if val {
    fmt.Printf(" 1")
  } else {
    fmt.Printf(" 0")
  }
}

func intFormat(maxInt int) string {
  if maxInt <= 9 {
    return "%d"
  }
  if maxInt <= 99 {
    return "%2d"
  }
  if maxInt <= 999 {
    return "%3d"
  }
  if maxInt <= 9999 {
    return "%4d"
  }
  return "%d"
}

func (df *DataFrame) debugPrint(context string) {
  if df.debug {
    log.Printf("%s [structure: %d data: %s dim: %dx%d allocated: %d]",
               context, df.structureUID, df.printableDataUID(), len(df.indices),
               df.NumColumns(), df.NumAllocatedRows())
  }
}
