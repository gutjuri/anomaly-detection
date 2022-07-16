package main

import (
	"bufio"
	"compress/gzip"
	"io"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

const outDir = "./"
const inDir = "/var/log/"

func fwriter(logQueue <-chan string, name string, wg *sync.WaitGroup) {
	wg.Add(1)
	hfile, err := os.OpenFile(outDir+name, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	if err != nil {
		log.Fatal(err)
	}
	defer hfile.Close()

	for {
		logLine, open := <-logQueue
		if !open {
			break
		}
		io.Copy(hfile, strings.NewReader(logLine+"\n"))
	}
	wg.Done()
}

var rex = regexp.MustCompile("[ ]+")

func processFile(name string, chans map[string]chan string, wg *sync.WaitGroup) {
	log.Println("Processing file " + name)
	fl, err := os.Open(name)
	if err != nil {
		log.Fatal(err)
	}
	defer fl.Close()

	var scanner *bufio.Scanner
	if strings.HasSuffix(name, ".gz") {
		freader, err := gzip.NewReader(fl)
		if err != nil {
			log.Fatal(err)
		}
		defer freader.Close()
		scanner = bufio.NewScanner(freader)
	} else {
		scanner = bufio.NewScanner(fl)
	}
	for scanner.Scan() {
		logLine := scanner.Text()
		fields := rex.Split(logLine, 6)
		node := fields[3]

		correctChannel, ok := chans[node]
		if !ok {
			correctChannel = make(chan string, 10000)
			chans[node] = correctChannel
			go fwriter(correctChannel, node+".log", wg)
		}

		correctChannel <- logLine
	}
}

func getMessageFiles() []string {
	fs, err := filepath.Glob(inDir + "messages-*.gz")
	if err != nil {
		log.Fatal(err)
	}
	sort.Strings(fs)
	return fs
}

func main() {
	chans := make(map[string]chan string)
	wg := new(sync.WaitGroup)

	for _, fl := range getMessageFiles() {
		processFile(fl, chans, wg)
	}

	for _, ch := range chans {
		close(ch)
	}
	wg.Wait()
}
