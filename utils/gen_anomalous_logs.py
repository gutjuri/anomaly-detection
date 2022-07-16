import random

random.seed(1337)
normal_logs_file = "drainres.log"
outFile = "drainres_abnormal.log"
outFileLabels = "labels.log"


def generate_randomly(normal_seqs, n_gen=10000):

    normal_seqs = list(map(lambda l: l.split("|"), normal_seqs))
    normal_seqs = list(
        map(lambda l: (int(l[0]), list(map(int, l[1].split(" ")[:-1]))), normal_seqs)
    )

    n_logkeys = max(map(max, map(lambda x: x[1], normal_seqs)))
    minLen = min(map(len, map(lambda x: x[1], normal_seqs)))
    maxLen = max(map(len, map(lambda x: x[1], normal_seqs)))
    ids = set(map(lambda x: x[0], normal_seqs))
    seqs = set(map(lambda x: tuple(x[1]), normal_seqs))

    print("Processed normal Seqs")
    anomalous_seqs = []
    for i in range(n_gen):
        seqid = gen_id(ids)
        seq = gen_seq(minLen, maxLen, seqs, n_logkeys)

        anomalous_seqs.append((seqid, seq))

        if i % 1000 == 0:
            print(f"Generated {i}/{n_gen}")


def printseq(idseq):
    id = idseq[0]
    seq = idseq[1]
    return f"{id}|{' '.join(map(str, seq))}\n"


def gen_id(ids):
    candidate = random.randrange(0, 10000000)
    while candidate in ids:
        candidate = random.randrange(0, 10000000)
    ids.add(candidate)
    return candidate


def gen_of_len(l, n_logkeys):
    seq = []
    for _ in range(l):
        seq.append(random.randrange(1, n_logkeys + 1))
    return seq


def gen_seq(minLen, maxLen, seqs, n_logkeys):
    seqlen = random.randrange(minLen, maxLen + 1)
    candidate = gen_of_len(seqlen, n_logkeys)
    while tuple(candidate) in seqs:
        candidate = gen_of_len(seqlen)
    seqs.add(tuple(candidate))
    return candidate


if __name__ == "__main__":
    f = open(normal_logs_file)
    normal_seqs = f.readlines()
    f.close()

    anomalous_seqs = generate_randomly(normal_seqs)
    print("Generated anomalous Seqs")
    f = open(outFile, "w")
    f.writelines(map(printseq, anomalous_seqs))
    f.close()

    print("Wrote Anomalous Seqs")

    f = open(outFileLabels, "w")
    f.write("id,label\n")
    for s in normal_seqs:
        f.write(f"{s[0]},Normal\n")
    for s in anomalous_seqs:
        f.write(f"{s[0]},Anomaly\n")
    f.close()

    print("Wrote label file")
