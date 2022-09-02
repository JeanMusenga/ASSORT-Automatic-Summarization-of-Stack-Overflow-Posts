import pickle
import pandas as pd
import torch
import io
import os
from google.colab import files
from tqdm import trange
import ast
import pickle
!pip install tqdm
!pip install transformers
!pip install torch
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
dataset =  pd.read_csv("dataset.csv")
summary_dict = dict()
from tqdm import trange
abstractive_summary	 = []
print("Getting abstractive summaries from BART-large-CNN")
for i in trange(len(dataset.answer_body)):
  i = dataset.answer_body[i]
  if i not in summary_dict:
    summary_dict[i] = summarizer(i[:3000], max_length=130, min_length=0, do_sample=False)[0]["summary_text"]
  abstractive_summary.append(summary_dict[i])
dataset["abstractive_summary"] = abstractive_summary

from transformers import pipeline
nli_pipe = pipeline(task="sentiment-analysis", model="roberta-large-mnli", return_all_scores=True, device=0)
def nli(premise, hypothesis):
  return nli_pipe("{} </s></s> {}".format(premise, hypothesis))[0][2]['score']

entailment2 = []
from tqdm import trange
print("Getting entailment scores from DocNLI model")
for index in trange(len(dataset['sentence'])):
  premise = dataset["abstractive_summary"][index]
  hypothesis = dataset['sentence'][index]
  entailment2.append(nli(premise, hypothesis))
dataset["entailment"] = entailment2

print("Splitting train/dev/test sets")
from sklearn.model_selection import train_test_split

print("Performing indirect supervision approach...No train data required")
print("Experiment with different thresholds...")

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

def cal_metrics(threshold):
  pred = [1 if i > threshold else 0 for i in dataset["entailment"]]
  true = list(dataset["truth"])
  precision = precision_score(true, pred)
  recall = recall_score(true, pred)
  f1 = 2 * precision * recall / (precision + recall)
  print("For threshold", threshold, "precision:", precision)
  print("For threshold", threshold, "recall:", recall)
  print("For threshold", threshold, "F1:", f1)
  print(" ")

for i in np.linspance(0, 1, 10):
  cal_metrics(i)