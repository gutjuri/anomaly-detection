package main

import (
	"bufio"
	"compress/gzip"
	"errors"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gitlab.uni-ulm.de/omi-teaching/thesis/msc-dispan/code/utils"
)

const (
	accountIx  = 0
	endTimeIx  = 26
	groupIx    = 30
	jobIdRawIx = 32
	nodeListIx = 56
	startIx    = 76
	userIx     = 102
)

func isComplete(jobInfo []string) bool {
	for _, x := range jobInfo {
		if x == "" || x == "Unknown" {
			return false
		}
	}
	return true
}

func parseNodeList(nodeList string) []string {
	ret := make([]string, 0)
	tokens := strings.Split(nodeList[2:len(nodeList)-1], ",")
	for _, token := range tokens {
		if strings.Contains(token, "-") {
			bounds := strings.Split(token, "-")
			lower, err := strconv.Atoi(bounds[0])
			if err != nil {
				log.Fatal(err)
			}
			upper, err := strconv.Atoi(bounds[1])
			if err != nil {
				log.Fatal(err)
			}
			for i := lower; i <= upper; i++ {
				ret = append(ret, "n"+fmt.Sprintf("%04d", i))
			}
		} else {
			ret = append(ret, "n"+token)
		}
	}
	return ret
}

type usageIv struct {
	startTime int
	endTime   int
	user      string
	id        int
}

func readSacct(inDir string) map[string][]usageIv {
	ret := make(map[string][]usageIv)
	inFiles, err := filepath.Glob(inDir + "sacct-*.gz")
	if err != nil {
		log.Fatal(err)
	}
	for _, path := range inFiles {
		fl, err := os.Open(path)
		if err != nil {
			log.Fatal(err)
		}
		defer fl.Close()
		freader, err := gzip.NewReader(fl)
		if err != nil {
			log.Fatal(err)
		}
		defer freader.Close()

		scanner := bufio.NewScanner(freader)
		noName := 0
		withName := 0
		scanner.Scan()
		for scanner.Scan() {
			logLine := scanner.Text()
			tokens := strings.Split(logLine, "|")
			interestingTokens := []string{
				tokens[userIx],
				tokens[groupIx],
				tokens[nodeListIx],
				tokens[startIx],
				tokens[endTimeIx],
			}
			if isComplete(interestingTokens) {
				withName++

				var nodes []string
				if strings.Contains(tokens[nodeListIx], ",") {
					nodes = parseNodeList(tokens[nodeListIx])
				} else {
					nodes = []string{tokens[nodeListIx]}
				}
				for _, node := range nodes {
					s, err := time.Parse("2006-01-02T15:04:05", tokens[startIx])
					if err != nil {
						log.Fatal(err)
					}
					e, err := time.Parse("2006-01-02T15:04:05", tokens[endTimeIx])
					if err != nil {
						log.Fatal(err)
					}
					jobId, err := strconv.Atoi(strings.SplitN(tokens[jobIdRawIx], ".", 2)[0])
					if err != nil {
						log.Fatal(err)
					}
					u := usageIv{
						startTime: int(s.Unix()),
						endTime:   int(e.Unix()),
						user:      tokens[userIx],
						id:        jobId,
					}
					ret[node] = append(ret[node], u)
					//fmt.Println(u, node)
				}
			} else {
				noName++
			}
		}
		log.Println(withName, noName, float64(withName)/(float64(withName)+float64(noName)))

	}
	return ret
}

func userAt(usageTimes []usageIv, ts int) (usageIv, error) {
	lower := 0
	upper := len(usageTimes) - 1
	steps := 0
	for {

		mid := (upper-lower)/2 + lower
		steps++
		if steps > 10000 {
			log.Fatal("Cannot complete binary search", lower, upper, mid, ts, usageTimes[mid], usageTimes[upper])
		}
		if usageTimes[mid].startTime <= ts && usageTimes[mid].endTime >= ts {
			return usageTimes[mid], nil
		}
		if lower >= upper {
			return usageIv{}, errors.New("not found")
		}
		if usageTimes[mid].startTime > ts {
			upper = mid - 1
		}
		if usageTimes[mid].endTime < ts {
			lower = mid + 1
		}

	}
}

// The following messages are routine messages and filtered out.
var exactIrrelevantMatches = []string{
	"Started LXC3 provision service.",
	"provision.service: Succeeded.",
	"Starting system activity accounting tool...",
	"sysstat-collect.service: Succeeded.",
	"Started system activity accounting tool.",
	"Starting update of the root trust anchor for DNSSEC validation in unbound...",
	"unbound-anchor.service: Succeeded.",
	"Started update of the root trust anchor for DNSSEC validation in unbound.",
	"Starting Generate a daily summary of process accounting...",
	"sysstat-summary.service: Succeeded.",
	"Started Generate a daily summary of process accounting.",
}

var irrelavantRegexes = []regexp.Regexp{
	*regexp.MustCompile("CPU[\\d]+: Package temperature above threshold, cpu clock throttled.*"),
	*regexp.MustCompile("CPU[\\d]+: Package temperature/speed normal"),
	*regexp.MustCompile("CPU[\\d]+: Core temperature above threshold, cpu clock throttled.*"),
	*regexp.MustCompile("CPU[\\d]+: Core temperature/speed normal"),
	*regexp.MustCompile("Selected source [\\d.]+"),
}

var startSkip = regexp.MustCompile("[a-zA-Z0-9._-]+ invoked oom-killer: .*")
var endSkip = regexp.MustCompile("oom_reaper: reaped process [\\d]+ .*")
var skippingNodes = make(map[string]interface{})

func isIrrelevant(node string, pname string, logContent string) bool {
	if strings.HasPrefix(pname, "stress-ng") {
		return true
	}
	for _, m := range exactIrrelevantMatches {
		if m == logContent {
			return true
		}
	}
	for _, m := range irrelavantRegexes {
		if m.MatchString(logContent) {
			return true
		}
	}

	if startSkip.MatchString(logContent) {
		skippingNodes[node] = struct{}{}
	}
	if endSkip.MatchString(logContent) {
		delete(skippingNodes, node)
		return true
	}
	if _, ok := skippingNodes[node]; ok {
		return true
	}

	return false
}

func processLogFile(inFile *os.File, outFile *os.File, usageTimes map[string][]usageIv) {
	scan, closer := utils.GetReader(inFile)
	defer closer()

	for scan.Scan() {
		logLine := scan.Text()
		fields := utils.ParseLine(logLine)
		node := fields[3]
		if isIrrelevant(node, fields[4], fields[5]) {
			continue
		}
		if uTimeNode, ok := usageTimes[node]; ok {
			ltime, err := utils.ParseMsgTime(fields)
			if err != nil {
				log.Fatal(err)
			}
			u, err := userAt(uTimeNode, int(ltime.Unix()))
			if err != nil {
				u.user = "Unknown"
				u.id = -1
			}
			if err == nil {
				fmt.Fprintf(outFile, "%v %v %v\n", u.user, u.id, logLine)
			}
		}
	}
}

func main() {
	inDir := os.Args[1]

	usageTimes := readSacct(inDir)
	inFiles, err := filepath.Glob(inDir + "messages-*.gz")
	if err != nil {
		log.Fatal(err)
	}

	for _, inPath := range inFiles {
		inFile, _ := os.Open(inPath)
		defer inFile.Close()
		processLogFile(inFile, os.Stdout, usageTimes)
	}

}
