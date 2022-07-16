import sys
import matplotlib.pyplot as plt
from .utils import read_sacct

group1 = "group1"
group2 = "group2"
group3 = "group3"
outfile = "../Graphics/ex-groups.svg"


def delZeroes(x, labels):
    ret = x
    for l in labels:
        ret = ret[ret[l] != 0]
    return ret


# load the dataset
data = read_sacct(sys.argv[1], 1_000_000, False)

print("File read")
groups = data["Account"].unique()
grouped_data = {}
for group in groups:
    grouped_data[group] = data[data["Account"] == group]
    grouped_data[group].drop(["JobID", "Account"], axis=1, inplace=True)



data = delZeroes(data, ["AveCPU", "AveVMSize"])

data1 = data[(data["Account"] == group1)]
data2 = data[(data["Account"] == group2)]
data3 = data[(data["Account"] == group3)]


xlabel = "AveCPU"
ylabel = "AveVMSize"

b1 = plt.scatter(data1[xlabel], data1[ylabel], c="blue", marker='o', s=20)
b2 = plt.scatter(data2[xlabel], data2[ylabel], c="red", marker='x', s=20)
b3 = plt.scatter(data3[xlabel], data3[ylabel], c="green", marker='^', s=20)
plt.legend(
    [b1, b2, b3],
    ["Group 1", "Group 2", "Group 3"],
    loc="upper right",
)
print(data1["User"].unique())

plt.axis("tight")
plt.ylim((0, 2.5e11))
plt.xlim((0, 4e6))
plt.xlabel(xlabel + " [s]")
plt.ylabel(ylabel + " [B]")

#plt.show()
plt.savefig(outfile)
