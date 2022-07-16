package utils

import (
	"bufio"
	"compress/gzip"
	"encoding/csv"
	"log"
	"os"
	"regexp"
	"strings"
	"time"
)

var rex = regexp.MustCompile("[ ]+")

func ParseLine(line string) []string {
	return rex.Split(line, 6)
}

func ParseMsgTime(fields []string) (time.Time, error) {
	return time.Parse("2006 Jan 2 15:04:05", "2022 "+fields[0]+" "+fields[1]+" "+fields[2])
}

func GetReader(inFile *os.File) (*bufio.Scanner, func() error) {
	var scan *bufio.Scanner
	if strings.HasSuffix(inFile.Name(), ".gz") {
		freader, err := gzip.NewReader(inFile)
		if err != nil {
			log.Fatal(err)
		}
		scan = bufio.NewScanner(freader)
		return scan, freader.Close
	} else {
		scan = bufio.NewScanner(inFile)
		return scan, func() error { return nil }
	}

}

func GetCSVReader(inFile *os.File) *csv.Reader {
	var rd *csv.Reader
	if strings.HasSuffix(inFile.Name(), ".gz") {
		freader, err := gzip.NewReader(inFile)
		if err != nil {
			log.Fatal(err)
		}
		rd = csv.NewReader(freader)
	} else {
		rd = csv.NewReader(inFile)
	}
	rd.Comma = '|'
	return rd
}

// Apply a function to each element of a slice.
func Apply[T any, R any](slice []T, f func(T) R) []R {
	res := make([]R, 0, len(slice))
	for _, elem := range slice {
		res = append(res, f(elem))
	}
	return res
}

// Tally Elements of a slice
func Count[T comparable](slice []T) map[T]int {
	ret := make(map[T]int, 0)
	for _, elem := range slice {
		ret[elem]++
	}
	return ret
}
