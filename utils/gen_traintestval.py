from sklearn.model_selection import train_test_split
import random

random.seed(1337)

TRAIN_P = 0.2
TEST_P = 0.5

f = open("drainres.log")
a_normal = f.readlines()
f.close()

f = open("drainres_abnormal.log")
a_abnormal = f.readlines()
f.close()

print("Loaded files")

x_train, a_normal = train_test_split(a_normal, train_size=TRAIN_P, random_state=1337)

f = open("x_train.log", "w")
f.writelines(x_train)
f.close()

print("Wrote x_train")

x_test, x_val = train_test_split(a_normal + a_abnormal, train_size=TEST_P, random_state=1337)

print(f"Training: {len(x_train)}, Test: {len(x_test)}, Val: {len(x_val)}")

f = open("x_test.log", "w")
f.writelines(x_test)
f.close()

f = open("x_val.log", "w")
f.writelines(x_val)
f.close()