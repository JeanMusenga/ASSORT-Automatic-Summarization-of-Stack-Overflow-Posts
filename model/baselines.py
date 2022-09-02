import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

# Evaluation
def calculate(truth, pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index, i in enumerate(truth):
        if i == 1 and pred[index] == 1:
            tp += 1
        elif i == 1 and pred[index] == 0:
            fn += 1
        elif i == 0 and pred[index] == 1:
            fp += 1
        else:
            tn += 1
    precision = float(tp) / (tp + fp)
    recall = float(tp) / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = float(tp + tn) / (tp + fp + tn + fn)
    print("F1: {}, Precision: {}, Recall: {}, Accuracy: {}".format(f1, precision, recall, accuracy))
    return f1, precision, recall, accuracy

from lexrank import STOPWORDS, LexRank
from path import Path

documents = []
documents_dir = Path('bbc/politics')
for file_path in documents_dir.files('*.txt'):
    with file_path.open(mode='rt', encoding='utf-8') as fp:
        documents.append(fp.readlines())

lxr = LexRank(documents, stopwords=STOPWORDS['en'])

pred = []
answer_body = [sent_tokenize(i) for i in dataset.answer_body]
for i in answer_body:
  summary = lxr.get_summary(answer_body, summary_size=5, threshold=.1)
  onetruth = [1 if j in summary else 0 for j in i]
  pred += onetruth

calculate(dataset.truth, pred)
