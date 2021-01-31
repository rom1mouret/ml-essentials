package preprocessing

import (
  "github.com/rom1mouret/ml-essentials/v0/dataframe"
)

type HashEncoderOptions struct {
  // nothing yet
}

// HashEncoder is a json-serializable structure that hashes strings into
// integers. The hashing algorithm always returns the same int if given the same
// string.
// Missing strings will be converted to -1.
// This preprocessor is primarily meant to be used as a first step before
// categorical encoding.
// Unless the number of categories is really huge, you need not worry about
// hashing collisions, especially if you run the encoder on a 64-bit system.
type HashEncoder struct {
  CategoricalColumns []string
}

// NewHashEncoder allocates a new HashEncoder.
func NewHashEncoder(opt HashEncoderOptions) *HashEncoder {
  return new(HashEncoder)
}

// Fit implements PreprocTraining and Transform interfaces.
func (encoder *HashEncoder) Fit(df *dataframe.DataFrame) error {
  encoder.CategoricalColumns = df.StringHeader().NameList()
  return nil
}

// TransformView implements PreprocTraining and Transform interfaces.
// This function is multi-threaded.
func (encoder *HashEncoder) TransformView(df *dataframe.DataFrame) (*dataframe.DataFrame, error) {
  return df.HashStringsView(encoder.CategoricalColumns...), nil
}

// TransformedColumns implements PreprocTraining and Transform interfaces.
func (encoder *HashEncoder) TransformedColumns() []string {
  return encoder.CategoricalColumns
}
