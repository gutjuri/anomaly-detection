from datetime import timedelta
import re
from typing import List
import pandas as pd
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
)

# Columns in Slurm output that we're interested in
cols = [
    "Account",
    "NCPUS",
    "NNodes",
    "NTasks",
    "AllocTRES",
    "AveCPU",
    "AveCPUFreq",
    "AveDiskRead",
    "AveDiskWrite",
    "AvePages",
    "AveRSS",
    "AveVMSize",
    "ConsumedEnergyRaw",
    "CPUTimeRAW",
    "ElapsedRaw",
    "MaxDiskRead",
    "MaxDiskWrite",
    "MaxPages",
    "MaxRSS",
    "MaxVMSize",
    "MinCPU",
    "TimelimitRaw",
    "User",
    "JobID",
]


def getgpus(tres: str):
    if type(tres) != str:
        return 0
    for kv in tres.split(","):
        if kv.startswith("gres/gpu="):
            return int(kv.split("=")[1])
    return 0


regex = re.compile(
    r"((?P<days>\d+?)-)?((?P<hours>\d+?):)((?P<minutes>\d+?):)((?P<seconds>\d+?))"
)


def parse_time(time_str):
    if type(time_str) != str:
        return 0

    parts = regex.match(time_str)
    if not parts:
        raise ValueError("no valid time: " + time_str)
    parts = parts.groupdict()
    time_params = {}
    for (name, param) in parts.items():
        if param:
            time_params[name] = int(param)
    return timedelta(**time_params).total_seconds()


n_rex = re.compile(r"(?P<n>\d+(.\d+)?)")


def parse_h(s):
    if type(s) != str:
        return 0
    parts = n_rex.match(s)
    num = float(parts.groupdict()["n"])
    lastc = s[-1]
    if lastc.isnumeric():
        return num
    if lastc == "K":
        return num * 1_000
    if lastc == "M":
        return num * 1_000_000
    if lastc == "G":
        return num * 1_000_000_000
    raise ValueError("Unit " + lastc + " not supported.")


def prep_jobs(jobs_raw: pd.DataFrame):
    # jobs_raw["job"] = jobs_raw["JobID"].apply(lambda id: id[:7])
    jobs = jobs_raw[jobs_raw["JobID"].apply(lambda s: "." in s)]
    jobs["interactive"] = jobs["JobID"].apply(
        lambda s: 1 if "interactive" in s else 0)

    timecols = ["AveCPU", "MinCPU"]
    hcols = [
        "AveCPUFreq",
        "AveDiskRead",
        "AveDiskWrite",
        "AveRSS",
        "AveVMSize",
        "AvePages",
        "MaxDiskRead",
        "MaxDiskWrite",
        "MaxRSS",
        "MaxVMSize",
        "MaxPages",
    ]
    # print(jobs.dtypes)
    for col in timecols:
        jobs[col] = jobs[col].apply(parse_time)
    for col in hcols:
        jobs[col] = jobs[col].apply(parse_h)

    jobs.fillna(value=0.0, inplace=True)

    return jobs


def read_sacct(path: str, nrows: int, drop_users: bool = True) -> pd.DataFrame:
    data = pd.read_csv(
        path,
        delimiter="|",
        engine="c",
        nrows=nrows,
        memory_map=True,
        usecols=cols,
        low_memory=False,
        encoding_errors='backslashreplace',
    )
    data["gpu"] = data["AllocTRES"].apply(getgpus)
    if drop_users:
        data.drop(["AllocTRES", "User"], axis=1, inplace=True)
    data = prep_jobs(data)
    return data

# Repeat the elements in vals.
# Results in a list of length n.


def iterate(n: int, vals: List):
    ret = []
    for i in range(n):
        ret.append(vals[i % len(vals)])
    return ret


def calc_metrics(res):
    # print(res)
    prec, recall, f1, _ = precision_recall_fscore_support(
        res["y_true"], res["y_test"], average="binary", pos_label=1
    )
    cm = confusion_matrix(res["y_true"], res["y_test"])
    # print(cm)
    return {
        # "FP": FP,
        "Precision": prec,
        "Recall": recall,
        "F-measure": f1,
        "cm": cm,
        "y_pred": res["y_test"],
        "y_true": res["y_true"],
    }
