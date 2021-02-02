package utils

import "math/rand"

type ProcessedJob struct {
  Key     string
  Error   error
  Result  interface{}
}

type StringQ struct {
  Keys    chan string
  Errors  chan ProcessedJob
  Workers int
  NumJobs int
}

func CreateStringQueue(keys []string, maxWorkers int) StringQ {
  if len(keys) == 0 {
    maxWorkers = 1
  } else if len(keys) < maxWorkers {
    maxWorkers = len(keys)
  } else {
    // shuffle the key list to avoid processing harder columms last
    a := make([]string, len(keys))
    copy(a, keys)
    rand.Shuffle(len(a), func(i, j int) { a[i], a[j] = a[j], a[i] })
    keys = a
  }
  result := StringQ{
      Keys: make(chan string, len(keys)),
      Errors:  make(chan ProcessedJob, len(keys)),
      Workers: maxWorkers,
      NumJobs: len(keys),
  }
  defer close(result.Keys)
  // push the job to the channel
  for _, col := range keys {
      result.Keys <- col
  }
  return result
}

func (q StringQ) Results() []ProcessedJob {
  // wait for all results to be pushed
  defer close(q.Errors)
  results := make([]ProcessedJob, q.NumJobs)
  for i := range results {
    results[i] = <- q.Errors
  }
  return results
}

func (q StringQ) Wait() {
  defer close(q.Errors)
  for i := 0; i < q.NumJobs; i++ {
      <- q.Errors
  }
}

func (q StringQ) Next() string {
  col, more := <-q.Keys
  if more {
    return col
  }
  return ""
}

func (q StringQ) Notify(err ProcessedJob) {
  q.Errors <- err
}
