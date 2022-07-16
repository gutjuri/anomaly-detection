import json
import random
import sys
import time
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline, Pipeline
from datetime import datetime
from collections import defaultdict
import pandas as pd
import seaborn as sbs
from .utils import iterate, calc_metrics

# np.set_printoptions(threshold=sys.maxsize)
seed = 1337
rng = random.Random(seed)

MIN_LOGINS = 100
CM_PATH = f"../Graphics/lo-all-3.svg"
TRAIN_SIZE = 0.75
VALIDATION_SIZE = 0.5
ANOMALY_SIZE = 0.15
HYPERPARAMS = False
RANDOM_TESTDATA = True

l_tr = 0
l_te = 0
l_val = 0

f = open("login_to_vec/AS553.json")
asdata = json.load(f)
f.close()
prefixes = list(map(lambda x: x["name"], asdata["prefixes"]))
prefixes += list(map(lambda x: x["name"], asdata["prefixes6"]))
prefixes += ["Internal", "Other"]
prefixes = list(set(prefixes))

print(f"There are a total of {len(prefixes)} possible source locations.")


def gen_anomalous_logins(n, username, behaviour):
    anomalous_logins = []
    anomalous_institutions = prefixes[:]
    anomalous_authmethods = ["keyboard-interactive/pam", "publickey"]
    anomalous_days = list(range(7))
    anomalous_hours = list(range(32))
    for p in behaviour:
        try:
            anomalous_institutions.remove(p[0])
        except ValueError:
            pass
        try:
            pass
            # anomalous_authmethods.remove(p[1])
        except ValueError:
            pass
        try:
            anomalous_days.remove(p[2])
        except ValueError:
            pass
        try:
            anomalous_hours.remove(p[3])
        except ValueError:
            pass

    if anomalous_institutions == []:
        return []
        anomalous_institutions = prefixes[:]
    if anomalous_authmethods == []:
        return []
        anomalous_authmethods = ["keyboard-interactive/pam", "publickey"]
    if anomalous_days == []:
        return []
        anomalous_days = list(range(7))
    if anomalous_hours == []:
        return []
        anomalous_hours = list(range(32))
    for x in range(n):
        anomalous_logins.append(
            [
                anomalous_institutions[rng.randrange(
                    len(anomalous_institutions))],
                anomalous_authmethods[rng.randrange(
                    len(anomalous_authmethods))],
                anomalous_days[rng.randrange(len(anomalous_days))],
                anomalous_hours[rng.randrange(len(anomalous_hours))],
            ]
        )
    return anomalous_logins


def to_grey(n):
    return n ^ (n >> 1)


def transform_X(X):
    time = X[2]
    dt = datetime.strptime(time, "%Y-%m-%dT%H:%M:%SZ")
    time_of_day = (dt.hour * 60 * 60 + dt.minute *
                   60 + dt.second) / (60 * 60 * 24) * 32
    # print(dt, time_of_day)
    X[2] = dt.weekday()
    X = np.append(X, to_grey(int(time_of_day)))
    # print(X)
    return X


# Get a One-Hot encoder that encodes a vector (location, authmethod, day-of-week)
def get_onehot_encoder(mode="OHE"):
    fitdata = np.array(prefixes).reshape((-1, 1))

    fitdata = np.append(
        fitdata,
        iterate(
            len(prefixes), [["keyboard-interactive/pam"],
                            ["publickey"], ["hostbased"]]
        ),
        axis=1,
    )
    fitdata = np.append(
        fitdata, iterate(len(prefixes), list(map(lambda x: [x], range(7)))), axis=1
    )

    if mode == "OHE":
        ohe = OneHotEncoder(sparse=False)
        ohe.fit(fitdata)
        return ohe
    else:
        ohe = OrdinalEncoder()
        ohe.fit(fitdata)
        return ohe


def to_bin(X):
    ret = np.zeros((X.shape[0], 5))
    # print(X.shape)
    for i in range(X.shape[0]):
        bin = f"{X[i][0]:05b}"
        for j in range(5):
            if bin[j] == "1":
                ret[i][j] = 1
    return ret


# Convert a vector of binary-encoded values
# to a vector of values.
def from_bin(X):
    ret = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        sm = 0
        for j in range(5):
            sm += X[i][4 - j] * (2**j)
        ret[i] = sm
    return ret


# Encode a input vector.
# All fields (except the last three fields) are one-hot encoded.
# The last three fields are converted to binary vectors.
def encode(ohe, X):
    X_t = ohe.transform(X[:, :3])
    return np.append(X_t, to_bin(X[:, 3:].astype(int)), axis=1)


def decode(ohe, X):
    X_t = ohe.inverse_transform(X[:, :-5])
    return np.append(X_t, from_bin(X[:, -5:]), axis=1)


def json_to_df(js):
    data = []
    for date in js:
        d = [date["Place"], date["AuthMethod"], date["Time"]]
        d = transform_X(d)
        data.append(d)
    return np.array(data)


def print_results(dc):
    global l_tr, l_te, l_val
    print(f"{'Name':17}Prec\tRecall\tfScore\ttFit\ttPred")
    res = []
    for k, v in dc.items():
        m = calc_metrics(v)
        b = {**v, **m}
        res.append((k, b))
    fig = plt.figure()
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.5)
    ax = gs.subplots()

    i = 0

    for x in sorted(res, key=lambda y: y[1]["F-measure"]):
        m = x[1]

        print(
            # f"{x[0]:17}{m['Precision']:.3f}\t{m['Recall']:.3f}\t{m['F-measure']:.3f}\t{m['t_fit']:.3f}\t{m['t_predict']:.3f}"
            f"{x[0]:17}&${m['cm'][1][1]}$&${m['cm'][0][1]}$&${m['cm'][0][0]}$&${m['cm'][1][0]}$??".replace(
                "?", "\\"
            )
        )

    for x in sorted(res, key=lambda y: y[1]["F-measure"]):
        m = x[1]

        print(
            # f"{x[0]:17}{m['Precision']:.3f}\t{m['Recall']:.3f}\t{m['F-measure']:.3f}\t{m['t_fit']:.3f}\t{m['t_predict']:.3f}"
            f"{x[0]:17}&${m['Precision']:.3f}$&${m['Recall']:.3f}$&${m['F-measure']:.3f}$&{'?SI{'}{1000*m['t_fit']/l_tr:.3f}{'}{?milli?second}'}&{'?SI{'}{1000*m['t_predict']/(l_te + l_val):.3f}{'}{?milli?second}??'}".replace(
                "?", "\\"
            )
        )

        # Only include these four algorithms in the confusion matrix
        if x[0] not in ["LOF-hamming", "iForest", "OCSVM", "OCSVM-SGD"]:
            continue

        axis = ax[i % 2, i // 2]
        axis.tick_params(labelsize=8)
        axis.tick_params(axis="y", labelrotation=45)
        disp = ConfusionMatrixDisplay.from_predictions(
            m["y_true"],
            m["y_pred"],
            normalize="pred",
            ax=axis,
            display_labels=["Normal", "Abnormal"],
            colorbar=False,
            # cmap="plasma"
            values_format=".4f",
        )
        axis.set_title(x[0])
        disp.im_.set_clim(0, 1)
        i += 1

    fig.colorbar(disp.im_, ax=ax)
    plt.savefig(CM_PATH)

    plt.clf()
    df = pd.DataFrame(
        {
            "Anomaly Score": 1 / (np.exp(dc["OCSVM-SGD"]["ascores"]) + 1),
            "Anomaly Score Raw": dc["OCSVM-SGD"]["ascores"],
            "Type": np.array(
                ["Anomaly" if x == 1 else "Normal" for x in dc["OCSVM-SGD"]["y_true"]]
            ),
            "Classified as": np.array(
                ["Anomaly" if x == 1 else "Normal" for x in dc["OCSVM-SGD"]["y_test"]]
            ),
        }
    )
    df[df["Type"] == "Normal"].sample(n=20).to_csv(
        "../Graphics/loginscores_normal.csv")
    df[df["Type"] == "Anomaly"].sample(n=20).to_csv(
        "../Graphics/loginscores_abnormal.csv"
    )


def sample(A: np.ndarray, n: int) -> np.ndarray:
    return A[np.random.choice(A.shape[0], n, replace=False)]


def evaluate_for_user(name):
    global l_tr, l_te, l_val

    logins_df = dataJson[name]

    dates = json_to_df(logins_df)

    ohe = get_onehot_encoder(mode="OHE")

    X = encode(ohe, dates)
    Y = np.zeros(len(X))

    X_training, X_test, Y_training, Y_test = train_test_split(
        X, Y, train_size=TRAIN_SIZE, random_state=seed
    )
    if RANDOM_TESTDATA:
        anomalous_logins = gen_anomalous_logins(
            int(len(X_test) * ANOMALY_SIZE), name, dates
        )
        if anomalous_logins == []:
            return {}
    else:
        anomalous_logins = np.concatenate(
            list(
                dict(filter(lambda v: v[0] != name, dataJson.items())).values()),
            axis=0,
        )
        anomalous_logins = sample(
            anomalous_logins, int(len(X_test) * ANOMALY_SIZE))
        anomalous_logins = json_to_df(anomalous_logins)
    anomalous_logins = encode(ohe, np.array(anomalous_logins))

    Y_test = np.append(Y_test, np.ones(len(anomalous_logins)))
    X_test = np.append(X_test, anomalous_logins, axis=0)

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, train_size=VALIDATION_SIZE, random_state=seed
    )

    l_tr += len(X_training)
    l_te += len(X_test)
    l_val += len(X_val)

    res = {}

    if HYPERPARAMS:
        # IF
        for n_est in [10, 20, 30, 50, 100, 200, 500, 1000]:
            for cont in [0.1, 0.2, 0.3, 0.4, 0.5]:
                res[f"iForest-{n_est}-{cont}"] = bm_classifier(
                    f"iForest-{n_est}-{cont}",
                    IsolationForest(
                        n_estimators=n_est, contamination=cont, random_state=seed
                    ),
                    X_training,
                    X_test,
                    Y_training,
                    Y_test,
                )
        # SVM
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for gamma in ["scale", "auto", 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

                    res[f"OCSVM-{kernel}-{gamma}-{nu}"] = bm_classifier(
                        f"OCSVM-{kernel}-{gamma}-{nu}",
                        svm.OneClassSVM(kernel=kernel, gamma=gamma, nu=nu),
                        X_training,
                        X_test,
                        Y_training,
                        Y_test,
                    )

        # SVM (SGD)
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for gamma in [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    res[f"OCSVM-SGD-{kernel}-{gamma}-{nu}"] = bm_classifier(
                        f"OCSVM-SGD-{kernel}-{gamma}-{nu}",
                        make_pipeline(
                            Nystroem(
                                gamma=gamma,
                                random_state=seed,
                                n_components=min(50, len(X_training)),
                                kernel=kernel,
                            ),
                            SGDOneClassSVM(
                                nu=nu,
                                shuffle=True,
                                fit_intercept=True,
                                average=True,
                                random_state=seed,
                                tol=1e-6,
                            ),
                        ),
                        X_training,
                        X_test,
                        Y_training,
                        Y_test,
                    )

        # OCSVM-SGD-retrain
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for gamma in [0.1, 0.2, 0.3, 0.4, 0.6, 0.8]:
                for nu in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    print(f"{kernel} {gamma} {nu} {len(X_training)}",
                          file=sys.stderr)
                    res[
                        f"OCSVM-SGD-retrain-{kernel}-{gamma}-{nu}"
                    ] = bm_classifier_adaptive(
                        f"OCSVM-SGD-retrain-{kernel}-{gamma}-{nu}",
                        Pipeline(
                            [
                                (
                                    "ns",
                                    Nystroem(
                                        gamma=gamma,
                                        random_state=seed,
                                        n_components=min(50, len(X_training)),
                                        kernel=kernel,
                                    ),
                                ),
                                (
                                    "svm",
                                    SGDOneClassSVM(
                                        nu=nu,
                                        shuffle=True,
                                        fit_intercept=True,
                                        average=True,
                                        random_state=seed,
                                        tol=1e-6,
                                    ),
                                ),
                            ],
                        ),
                        X_training,
                        X_test,
                        Y_training,
                        Y_test,
                    )
    else:
        res["iForest"] = bm_classifier(
            "iForest",
            IsolationForest(random_state=seed, n_estimators=50),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        res[f"OCSVM"] = bm_classifier(
            f"OCSVM",
            svm.OneClassSVM(kernel="poly", gamma=0.8, nu=0.1),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        res[f"OCSVM-SGD"] = bm_classifier(
            f"OCSVM-SGD",
            make_pipeline(
                Nystroem(
                    gamma=0.1,
                    random_state=seed,
                    n_components=min(50, len(X_training)),
                    kernel="rbf",
                ),
                SGDOneClassSVM(
                    nu=0.1,
                    shuffle=True,
                    fit_intercept=True,
                    average=True,
                    random_state=seed,
                    tol=1e-6,
                ),
            ),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        res["OCSVM-SGD-retrain"] = bm_classifier_adaptive(
            f"OCSVM-SGD-retrain",
            Pipeline(
                [
                    (
                        "ns",
                        Nystroem(
                            gamma=0.2,
                            random_state=seed,
                            n_components=min(50, len(X_training)),
                            kernel="rbf",
                        ),
                    ),
                    (
                        "svm",
                        SGDOneClassSVM(
                            nu=0.2,
                            shuffle=True,
                            fit_intercept=True,
                            average=True,
                            random_state=seed,
                            tol=1e-6,
                        ),
                    ),
                ],
            ),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        for dm in [
            "cosine",
            "braycurtis",
            "chebyshev",
            "correlation",
            "hamming",
            "minkowski",
        ]:
            res[f"LOF-{dm}"] = bm_classifier(
                f"LOF-{dm}",
                LocalOutlierFactor(
                    novelty=True, n_jobs=1, metric=dm, contamination=VALIDATION_SIZE
                ),
                X_training,
                X_val,
                Y_training,
                Y_val,
            )
    return res


# load the dataset
f = open("usual_behaviour.json")
dataJson = json.load(f)
f.close()


# Benchmark a given classifier
def bm_classifier(name, clf, X_training, X_test, Y_training, Y_test):
    t_start1 = time.time()
    clf.fit(X_training)
    t_end1 = time.time()

    t_start2 = time.time()
    X_test_res = clf.predict(X_test)
    t_end2 = time.time()

    X_test_res[X_test_res == 1] = 0
    X_test_res[X_test_res == -1] = 1

    return {
        "t_fit": t_end1 - t_start1,
        "t_predict": t_end2 - t_start2,
        "y_test": X_test_res,
        "y_true": Y_test,
        "ascores": clf.decision_function(X_test),
    }


def initial_pipe_fit(pipeline_obj, X):
    X = pipeline_obj.named_steps["ns"].fit_transform(X)
    pipeline_obj.named_steps["svm"].partial_fit(X)


def partial_pipe_fit(pipeline_obj, X, Xt):
    print(f"fit size: {Xt.shape}", file=sys.stderr)
    pipeline_obj.named_steps["ns"].fit(Xt)
    X = pipeline_obj.named_steps["ns"].transform(X)
    pipeline_obj.named_steps["svm"].partial_fit(X)


def bm_classifier_adaptive(name, clf, X_training, X_test, Y_training, Y_test):
    t_start1 = time.time()
    initial_pipe_fit(clf, X_training)
    t_end1 = time.time()

    X_test_res = np.empty_like(Y_test)
    t_start2 = time.time()
    Xt = np.copy(X_training)
    for i, p in enumerate(X_test):
        res = clf.predict(p.reshape(1, -1))
        X_test_res[i] = res[0]
        if res[0] == -1 and Y_test[i] == 0:
            Xt = np.append(Xt, p.reshape(1, -1), axis=0)
            partial_pipe_fit(clf, p.reshape(1, -1), Xt)
            # print("ok")
            # if (clf.predict(p.reshape(1, -1))[0] == -1):
            #    print("Wft")
            # print("rr")

    t_end2 = time.time()

    X_test_res[X_test_res == 1] = 0
    X_test_res[X_test_res == -1] = 1

    return {
        "t_fit": t_end1 - t_start1,
        "t_predict": t_end2 - t_start2,
        "y_test": X_test_res,
        "y_true": Y_test,
        "ascores": clf.decision_function(X_test),
    }


res = defaultdict(lambda: {})
# Determine which user we want to inspect
users = list(dataJson.keys())
users.sort(key=lambda x: len(dataJson[x]), reverse=True)
testuser = users[0]

processedLogins = 0
processedUsers = 0
for user in users:
    if len(dataJson[user]) >= MIN_LOGINS:
        print(user, len(dataJson[user]))
        processedLogins += len(dataJson[user])
        processedUsers += 1
        r = evaluate_for_user(user)
        if r == {}:
            print("Warning: not able to generate anomalous data for " + user)
        for mod in r:
            for k in r[mod]:
                if k in ["y_test", "y_true", "ascores"]:
                    res[mod][k] = np.append(
                        res[mod].get(k, np.empty(0)), r[mod][k])
                else:
                    res[mod][k] = res[mod].get(k, 0) + r[mod][k]
# print(res)
print_results(res)
print("Processed logins: ", processedLogins)
print("Processed users: ", processedUsers)
print("Training data: ", l_tr)
print("Test data:", l_te)
print("Validation data:", l_val)
