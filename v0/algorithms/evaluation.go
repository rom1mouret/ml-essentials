package algorithms

import (
  "math"
  "fmt"
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
)

// MSE returns the mean-squared-error between two views on float vectors.
// mse = MEAN((yPred - yTrue)^2)
func MSE(yPred dataframe.FloatAccess, yTrue dataframe.FloatAccess) float64 {
  if yPred.Size() != yTrue.Size() {
    panic(fmt.Sprintf("size(yPred)=%d != size(yTrue)=%d", yPred.Size(), yTrue.Size()))
  }
  result := 0.0
  for i := 0; i < yPred.Size(); i++ {
    diff := yPred.Get(i) - yTrue.Get(i)
    result += diff * diff
  }
  return result / float64(yPred.Size())
}

// MAE returns the mean-absolute-error between two views on float vectors.
// mae = MEAN((|yPred - yTrue|)
func MAE(yPred dataframe.FloatAccess, yTrue dataframe.FloatAccess) float64 {
  if yPred.Size() != yTrue.Size() {
    panic(fmt.Sprintf("size(yPred)=%d != size(yTrue)=%d", yPred.Size(), yTrue.Size()))
  }
  result := 0.0
  for i := 0; i < yPred.Size(); i++ {
    result += math.Abs(yPred.Get(i) - yTrue.Get(i))
  }
  return result / float64(yPred.Size())
}

// TODO: Cross Entropy etc.
