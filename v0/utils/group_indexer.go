package utils

import "fmt"

type GroupIndexer struct {
  NGroups    int
  groupSize  int
  elementIdx int
  end        int
  groupIdx   int
}

func CreateGroupIndexer(nElements int, nGroups int) GroupIndexer {
  var indexer GroupIndexer
  indexer.Initialize(nElements, nGroups)
  return indexer
}

func (indexer *GroupIndexer) Initialize(nElements int, nGroups int) {
  if nGroups < 0 {
    panic(fmt.Sprintf("nGroups=%d is not a valid number", nGroups))
  }
  if nGroups > nElements {
    nGroups = nElements
  }
  indexer.NGroups = nGroups
  indexer.groupSize = nElements / nGroups
  if nElements % nGroups != 0 {
    indexer.groupSize++
  }
  indexer.end = nElements
}

func (indexer GroupIndexer) HasNext() bool {
  return indexer.elementIdx < indexer.end
}

func (indexer *GroupIndexer) Next() (int, int, int) {
  current := indexer.elementIdx
  nextIdx := current + indexer.groupSize
  if nextIdx > indexer.end {
    nextIdx = indexer.end
  }
  indexer.elementIdx = nextIdx
  indexer.groupIdx++

  return indexer.groupIdx-1, current, nextIdx
}
