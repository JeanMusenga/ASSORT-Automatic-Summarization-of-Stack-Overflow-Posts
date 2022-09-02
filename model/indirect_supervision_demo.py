import pickle
import pandas as pd
import torch
import io
import os
from google.colab import files
from tqdm import trange
import ast
import pickle
from transformers import pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

inputfile = ''
outputfile = ''
post = ""

# sentence tokenize it
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize
sentences = sent_tokenize(post)

# abstractive summary
abstractive_summary = summarizer(i[:3000], max_length=130, min_length=0, do_sample=False)[0]["summary_text"]

# nli
from transformers import pipeline
nli_pipe = pipeline(task="sentiment-analysis", model="roberta-large-mnli", return_all_scores=True, device=0)
def nli(premise, hypothesis):
  return nli_pipe("{} </s></s> {}".format(premise, hypothesis))[0][2]['score']

# output the result
result = ""
for i in sentences:
  if nli(abstractive_summary, i) > 0.33:
    result += i + " "

with open(outputfile, 'w') as f:
  f.write()
print("The result has been written to...", outputfile)