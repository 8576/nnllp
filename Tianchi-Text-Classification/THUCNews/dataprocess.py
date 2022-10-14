import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

SEED = 100
print(os.path.abspath("__file__"))
print(os.listdir("THUCNews"))


df = pd.read_csv(r"THUCNews\data\train_set.csv", sep="\t")
# print(df)
trainset = df.sample(frac=0.7, axis=0, random_state=SEED)
for idx in range(10):
    print("=" * 10 + str(idx) + str(trainset.iloc[idx].label) +"=" * 10)
    print(trainset.iloc[idx].text)
# print(trainset)
df = df[~df.index.isin(trainset.index)]
devset = df.sample(frac=0.63, axis=0, random_state=SEED)
testset = df[~df.index.isin(devset.index)]
print("=" * 5 + "trainset" + "=" * 5 )
print(trainset, devset.__len__())
# trainset.to_csv(os.path.join("THUCNews\data\\", "trainset.csv"), sep="\t", index=False)
print("=" * 5 + "devset" + "=" * 5 )
print(devset, devset.__len__())
# devset.to_csv(os.path.join("THUCNews\data\\", "devset.csv"), sep="\t", index=False)
print("=" * 5 + "testset" + "=" * 5 )
print(testset, testset.__len__())
# testset.to_csv(os.path.join("THUCNews\data\\", "testset.csv"), sep="\t", index=False)

print()
# word_count = dict()
# allwords = set()
# for text in train_df["text"]:
#     text = text.strip()
#     if text:
#         words = text.split(" ")
#         allwords.update(words)
#         length = len(words)
#         word_count[length]  = word_count.get(length, 0) + 1
# len_msg = sorted([ item for item in word_count.items()],  key=lambda x: x[1], reverse=True)
# len_list = list()
# for msg in len_msg:
#     # print(msg)
#     len_list.append(msg[0])
# print(len_msg[0], len_msg[-1])
# print(np.average(len_list), np.std(len_list), np.median(len_list), np.max(len_list))
# print(len(allwords))

