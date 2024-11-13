# First we import the necessary libraries
import torch
from src.data.assemblyLabelDataset import AssemblyLabelDataset

correct_dataset = AssemblyLabelDataset("./mistake_labels/", split="correct")
mistake_dataset = AssemblyLabelDataset("./mistake_labels/", split="mistake")

sample_len = 67
initial_padding = tuple([0]*sample_len)
final_padding = tuple([1]*sample_len)

all_samples = set()
for sample in correct_dataset:
    for array in sample["oh_sample"]:
        if tuple(array.tolist()) == final_padding:
            continue
        all_samples.add(tuple(array.tolist()))
for sample in mistake_dataset:
    for array in sample["oh_sample"]:
        if tuple(array.tolist()) == final_padding:
            continue
        all_samples.add(tuple(array.tolist()))


all_samples = tuple(all_samples)
all_samples = (initial_padding,) + all_samples

A = torch.zeros((len(all_samples), len(all_samples)))
for sample in correct_dataset:
    prev_step = initial_padding
    for n, array in enumerate(sample["oh_sample"]):
        if tuple(array.tolist()) == final_padding:
            continue
        A[all_samples.index(prev_step)][
            all_samples.index(tuple(array.tolist()))
        ] += 1
        prev_step = tuple(array.tolist())

threshold = 1/len(all_samples)

    

for n, line in enumerate(A):
    tot = sum(line)
    if tot > 0:
        A[n] = line / tot
    else:
        A[n] = torch.zeros(len(all_samples)) + threshold

labels = []
gt_labels = []

for sample in mistake_dataset:
    prev_step = initial_padding
    for n, array in enumerate(sample["oh_sample"]):
        if tuple(array.tolist()) == final_padding:
            continue
        p = A[all_samples.index(prev_step)][all_samples.index(tuple(array.tolist()))]
        labels.append(0) if p < threshold else labels.append(1)
        if int(tuple(sample["oh_label"].tolist()[n])[0]) != 1:
            gt_labels.append(0)
        elif int(tuple(sample["oh_label"].tolist()[n])[1]) != 1:
            gt_labels.append(1)
        else:
            gt_labels.append(1)
        prev_step = tuple(array.tolist())

# Calculate accuracy, precision, recall, F1
TP = 0
FP = 0
FN = 0
TN = 0
for n, label in enumerate(labels):
    if label == 1 and gt_labels[n] == 1:
        TP += 1
    elif label == 1 and gt_labels[n] == 0:
        FP += 1
    elif label == 0 and gt_labels[n] == 1:
        FN += 1
    elif label == 0 and gt_labels[n] == 0:
        TN += 1

accuracy = (TP + TN) / (TP + FP + FN + TN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * (precision * recall) / (precision + recall)

# Print results
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("F1: {}".format(F1))

print("TP: {}".format(TP))
print("FP: {}".format(FP))
print("FN: {}".format(FN))
print("TN: {}".format(TN))

# Accuracy: 0.675739247311828
# Precision: 0.7571277719112989
# Recall: 0.739556472408458
# F1: 0.7482389773023741
# TP: 1434
# FP: 460
# FN: 505
# TN: 577