import random
import sys
import time
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_recall_fscore_support,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from sklearn.kernel_approximation import Nystroem
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from datetime import timedelta
from collections import defaultdict
import pandas as pd
import seaborn as sbs
from .utils import read_sacct, calc_metrics

cap = None
head = 1_000_000
hyperparam = False
fname_out = f"../Graphics/aj-all-no-cap-3.svg"
corr_outname = f"../Graphics/aj-correlation-4.svg"
train_size = 0.1
test_size = 0.333

np.set_printoptions(threshold=sys.maxsize)
seed = 1337
np.random.seed(seed)
rng = random.Random(seed)

l_tr = 0
l_te = 0
l_val = 0


def print_results(dc):
    global l_tr, l_te, l_val
    print(f"{'Name':17}Prec\tRecall\tfScore\ttFit\ttPred")
    res = []
    for k, v in dc.items():
        m = calc_metrics(v)
        b = {**v, **m}
        res.append((k, b))
    fig = plt.figure()
    # fig.suptitle("haha")
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

        if x[0] not in ["LOF-braycurtis", "iForest", "OCSVM", "OCSVM-SGD"]:
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
    plt.savefig(fname_out)


def print_per_group(f1_per_group):
    plt.clf()

    res = {"Size of training data set": [], "F1-score": [], "Algorithm": []}
    for clf in ["OCSVM", "OCSVM-SGD", "iForest", "LOF-braycurtis"]:
        for i in range(len(f1_per_group[clf]["no_members"])):
            if f1_per_group[clf]["no_members"][i] < 10000:
                res["Size of training data set"].append(
                    f1_per_group[clf]["no_members"][i])
                res["F1-score"].append(f1_per_group[clf]["f1"][i])
                res["Algorithm"].append(clf)
    res = pd.DataFrame(res)
    ax = print(
        f"correlation: {res['F1-score'].corr(res['Size of training data set'])}")
    sbs.scatterplot(data=res, x="Size of training data set",
                    y="F1-score", hue="Algorithm")

    plt.savefig(corr_outname)


def sample(A: np.ndarray, n: int) -> np.ndarray:
    return A[np.random.choice(A.shape[0], n, replace=False), :]


def evaluate_for_group(
    name: str, X_dfs: Dict[str, np.ndarray], scaler: StandardScaler, tr: PCA
):
    global l_tr, l_te, l_val

    X_df = X_dfs[name]
    X = X_df

    Y = np.zeros(len(X))

    if cap != None and len(X) * train_size > cap:
        train_size_l = cap
    else:
        train_size_l = train_size

    X_training, X_test, Y_training, Y_test = train_test_split(
        X, Y, train_size=train_size_l, random_state=seed
    )

    anomalous_size = int(len(X_test) * 0.15)

    anomalous_logins = np.concatenate(
        list(dict(filter(lambda v: v[0] != name, X_dfs.items())).values()), axis=0
    )
    anomalous_logins = sample(anomalous_logins, anomalous_size)

    Y_test = np.append(Y_test, np.ones(len(anomalous_logins)))
    X_test = np.append(X_test, anomalous_logins, axis=0)

    X_val, X_test, Y_val, Y_test = train_test_split(
        X_test, Y_test, train_size=(1-test_size), random_state=seed
    )

    X_test = tr.transform(X_test)
    X_training = tr.transform(X_training)
    X_val = tr.transform(X_val)
    X_training = scaler.transform(X_training)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    l_tr += len(X_training)
    l_te += len(X_test)
    l_val += len(X_val)

    res = {}

    if hyperparam:
        # IF
        for n_est in [10, 20, 30, 50, 100]:
            # for cont in [0.1, 0.2]:
            res[f"iForest-{n_est}"] = bm_classifier(
                f"iForest-{n_est}",
                IsolationForest(
                    n_estimators=n_est, random_state=seed
                ),
                X_training,
                X_test,
                Y_training,
                Y_test,
            )
        # SVM
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for gamma in ["scale", "auto", 0.1, 0.2, 0.3, 0.4]:
                for nu in [0.1, 0.2, 0.3, 0.4]:

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
            for gamma in [0.1, 0.2, 0.3, 0.4]:
                for nu in [0.1, 0.2, 0.3, 0.4]:
                    res[f"OCSVM-SGD-{kernel}-{gamma}-{nu}"] = bm_classifier(
                        f"OCSVM-SGD-{kernel}-{gamma}-{nu}",
                        make_pipeline(
                            Nystroem(
                                gamma=gamma,
                                random_state=seed,
                                n_components=50,
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
        for dm in [
            "cosine",
            "braycurtis",
            "chebyshev",
            "correlation",
            "minkowski",
        ]:
            # for thr in [0.001, 0.01, 0.1, 0.2, 0.3, 'auto']:
            # print(dm)
            res[f"LOF-{dm}"] = bm_classifier(
                f"LOF-{dm}",
                LocalOutlierFactor(novelty=True, n_jobs=-1, metric=dm),
                X_training,
                X_val,
                Y_training,
                Y_val,
            )

    else:
        res["iForest"] = bm_classifier(
            "iForest",
            IsolationForest(random_state=seed, n_estimators=30),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        res[f"OCSVM"] = bm_classifier(
            f"OCSVM",
            svm.OneClassSVM(kernel="rbf", gamma=0.2, nu=0.1),
            X_training,
            X_val,
            Y_training,
            Y_val,
        )

        res[f"OCSVM-SGD"] = bm_classifier(
            f"OCSVM-SGD",
            make_pipeline(
                Nystroem(gamma=0.2, random_state=seed,
                         n_components=50, kernel="rbf"),
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

        for dm in [
            "cosine",
            "braycurtis",
            "chebyshev",
            "correlation",
            "minkowski",
        ]:
            # print(dm)
            res[f"LOF-{dm}"] = bm_classifier(
                f"LOF-{dm}",
                LocalOutlierFactor(novelty=True, n_jobs=-1, metric=dm),
                X_training,
                X_val,
                Y_training,
                Y_val,
            )
    return res


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
    }


# load the dataset
data: pd.DataFrame = None
for i in range(1, len(sys.argv)):
    fread = read_sacct(sys.argv[i], head)
    if data is not None:
        data = pd.concat([data, fread], axis=0)
    else:
        data = fread

groups = data["Account"].unique()
grouped_data = {}
for group in groups:
    grouped_data[group] = data[data["Account"] == group]
    grouped_data[group].drop(["JobID", "Account"], axis=1, inplace=True)
    # print(grouped_data[group][grouped_data[group].eq("2.50K").any(1)])
    grouped_data[group] = grouped_data[group].values.astype(np.float64)
    #print(group, len(grouped_data[group]))

res = defaultdict(lambda: {})

print(f"loaded {len(data)} jobs")

processedJobs = 0
processedUsers = 0

# Fit PCA model on the dataset
tr = PCA(n_components="mle").fit(
    np.concatenate(list(grouped_data.values()), axis=0))
print(
    f"n_components: {tr.n_components_} (before: {next(grouped_data.values().__iter__()).shape[1]})")

# Fit Scaler on the dataset
scaler = StandardScaler().fit(
    tr.transform(np.concatenate(list(grouped_data.values()), axis=0))
)

f1_per_group = defaultdict(lambda: {"no_members": [], "f1": []})

for group in groups:
    print(group, len(grouped_data[group]))

# Only consider groups with more than a minimum amount of jobs
groupsFiltered = list(filter(lambda g: len(
    grouped_data[g]) * train_size > 105, groups))
for i, group in enumerate(groupsFiltered):
    print(
        f"Processing group {i} of {len(groupsFiltered)}: {len(grouped_data[group])} jobs.")
    processedJobs += len(grouped_data[group])
    processedUsers += 1
    r = evaluate_for_group(group, grouped_data, scaler, tr)

    for mod in r:
        tmp = calc_metrics(r[mod])

        f1_per_group[mod]["no_members"].append(len(grouped_data[group]))
        # print(r[mod])
        f1_per_group[mod]["f1"].append(tmp["F-measure"])
        for k in r[mod]:
            if k in ["y_test", "y_true"]:
                res[mod][k] = np.append(
                    res[mod].get(k, np.empty(0)), r[mod][k])
            else:
                res[mod][k] = res[mod].get(k, 0) + r[mod][k]
# print(res)
print_results(res)
print_per_group(f1_per_group)
print("Processed jobs: ", processedJobs)
print("Processed groups: ", processedUsers)
print("Training data: ", l_tr)
print("Test data:", l_te)
print("Validation data:", l_val)
