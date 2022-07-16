package main

import (
	_ "embed"
	"encoding/json"
	"errors"
	"log"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"gitlab.uni-ulm.de/omi-teaching/thesis/msc-dispan/code/utils"
)

type asinfo struct {
	Asn       string   `json:"asn"`
	Name      string   `json:"name"`
	Country   string   `json:"country"`
	Allocated string   `json:"allocated"`
	Prefixes  []Prefix `json:"prefixes"`
	Prefixes6 []Prefix `json:"prefixes6"`
}

type Prefix struct {
	Netblock string `json:"netblock"`
	Id       string `json:"id"`
	Name     string `json:"name"`
}

func getOrg(ip string, prefixes []Prefix) int {
	ipAddr := net.ParseIP(ip)
	for i, pref := range prefixes {
		_, block, _ := net.ParseCIDR(pref.Netblock)
		if block.Contains(ipAddr) {
			return i
		}
	}
	return -1
}

var localRanges []*net.IPNet

func init() {
	locals := []string{"10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16"}
	for _, r := range locals {
		_, rng, _ := net.ParseCIDR(r)
		localRanges = append(localRanges, rng)
	}
}

func isInternalIp(ip string) bool {
	ipAddr := net.ParseIP(ip)
	for _, r := range localRanges {
		if r.Contains(ipAddr) {
			return true
		}
	}
	return false
}

type Login struct {
	User       string
	Success    bool
	SourceIp   string
	Place      string
	AuthMethod string
	Node       string
	Time       time.Time
}

var noLoginLine = errors.New("Not a login line")

func parseLogin(message []string, prefixes []Prefix) (Login, error) {
	ret := Login{}
	logLn := message[5]
	if !strings.HasPrefix(logLn, "Accepted") && !strings.HasPrefix(logLn, "Failed") {
		return ret, noLoginLine
	}

	fields := strings.Fields(logLn)
	if fields[2] != "for" || fields[3] == "invalid" {
		return ret, noLoginLine
	}
	if fields[0] == "Accepted" {
		ret.Success = true
	} else {
		ret.Success = false
	}
	ret.AuthMethod = fields[1]
	ret.User = fields[3]
	ret.SourceIp = fields[5]
	ret.Node = message[3]
	tm, err := utils.ParseMsgTime(message)
	if err != nil {
		log.Fatal(err)
	}
	ret.Time = tm
	placeId := getOrg(ret.SourceIp, prefixes)
	if placeId != -1 {
		ret.Place = prefixes[placeId].Name
	} else if isInternalIp(ret.SourceIp) {
		ret.Place = "Internal"
	} else {
		ip := net.ParseIP(ret.SourceIp)
		if strings.Contains(ret.SourceIp, ":") {
			// ipv6
			ret.Place = ip.Mask(net.CIDRMask(32, 128)).To16().String() + "/32"
		} else {
			ret.Place = ip.Mask(net.IPv4Mask(255, 255, 0, 0)).To4().String() + "/16"
		}
		ret.Place = "Other"

	}
	return ret, nil
}

//go:embed AS553.json
var asdataRaw []byte

func parsePid(line string) (int, []string, error) {
	logParsed := utils.ParseLine(line)
	// only process logs from login nodes
	if !strings.HasPrefix(logParsed[3], "login") {
		return 0, nil, errors.New("")
	}
	pidS := strings.FieldsFunc(logParsed[4], func(r rune) bool { return r == '[' || r == ']' })
	if len(pidS) < 2 {
		return 0, nil, errors.New("")
	}
	pid, err := strconv.Atoi(pidS[1])
	if err != nil {
		log.Fatal(err)
	}
	return pid, logParsed, nil

}

func learn(fnames []string, prefixes []Prefix) map[string][]Login {
	places := make(map[string][]Login)

	for _, fname := range fnames {
		fname, err := os.Open(fname)
		if err != nil {
			log.Fatal(err)
		}
		defer fname.Close()
		scan, closer := utils.GetReader(fname)
		defer closer()
		for scan.Scan() {
			line := scan.Text()
			_, logParsed, err := parsePid(line)
			if err != nil {
				continue
			}
			login, err := parseLogin(logParsed, prefixes)
			if err != nil || !login.Success {
				continue
			}
			places[login.User] = append(places[login.User], login)

		}
	}
	return places
}

func main() {
	asdata := asinfo{}
	err := json.Unmarshal(asdataRaw, &asdata)
	asdata.Prefixes = append(asdata.Prefixes, asdata.Prefixes6...)
	if err != nil {
		log.Fatal(err)
	}

	fnames := os.Args[1:]

	result := learn(fnames, asdata.Prefixes)
	js, err := json.Marshal(result)
	if err != nil {
		log.Fatal(err)
	}
	//storeFile, err := os.OpenFile("usual_behaviour.json", os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0666)
	storeFile := os.Stdout
	if err != nil {
		log.Fatal(err)
	}
	defer storeFile.Close()
	_, err = storeFile.Write(js)
	if err != nil {
		log.Fatal(err)
	}
	n := 0
	for _, v := range result {
		n += len(v)
	}
	log.Println("Processed ", n, " logins")

}
