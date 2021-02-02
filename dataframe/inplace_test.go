package dataframe

import (
    "testing"
    "golang.org/x/text/encoding/unicode"
    "golang.org/x/text/encoding/korean"
    u "github.com/rom1mouret/ml-essentials/utils"
)


func Test1Encode(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  builder.AddObjects("col", "고랭", "one", "two", nil).MarkAsString("col")
  df := fillBlanks(builder)
  ref := df.objects["col"][0].(string)
  df.Encode(unicode.UTF16(unicode.BigEndian, unicode.IgnoreBOM))
  u.AssertStringNotEquals("1st string", df.objects["col"][0].(string), ref, t)
  // convert back
  df.Encode(nil)
  u.AssertStringEquals("1st string", df.objects["col"][0].(string), ref, t)
}

func TestEncodeError(t *testing.T) {
  builder := DataBuilder{RawData: NewRawData()}
  notUTF8 := string([]byte{226, 28, 161, 0})
  builder.AddObjects("col", "고랭", "two", notUTF8).MarkAsString("col")
  df := fillBlanks(builder)
  err := df.Encode(korean.EUCKR)  // going from UTF8 to EUC-KR should fail
  if u.AssertTrue("error expected", err != nil, t) {
    u.AssertStringEquals("first string", df.objects["col"][0].(string), "고랭", t)
    u.AssertStringEquals("second string", df.objects["col"][1].(string), "two", t)
    u.AssertStringEquals("second string", df.objects["col"][2].(string), notUTF8, t)
  }
}
