from transformers import AutoModel, AutoTokenizer
import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
from classification_head import *


answer_body = ""
question_title = ""

# Load pretrained question classifier
print("Loading pre-trained question classifier...")
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
clf = pickle.load(open('question_classifier.sav', 'rb'))

print("Loading sentence-classifiers we just trained...")
# Howto model
size = 796
howtomodel = Net(size)
howtomodel.load_state_dict(torch.load('nn_old_approach_50_how_to_best.pt'))
# Conceptual model
conceptualmodel = Net(size)
conceptualmodel.load_state_dict(torch.load('nn_old_approach_50_conceptual_best.pt'))
# Debug model
debugmodel = Net(size)
debugmodel.load_state_dict(torch.load('nn_old_approach_50_debug_best.pt'))

print("Getting bag-of-word representations of question titles")
new_bag_of_words = vectorizer.transform([answer_body]).toarray()
probs = clf.predict_proba(new_bag_of_words)

print("Applying different sentence classifiers...")
def get_sm_from_sentence_classifier(model, embeddings):
  m = nn.Softmax(dim=1)
  scores = model(torch.tensor(embeddings).float())
  return m(scores).detach().numpy()[:, 1]

explicit_features_for_df = np.empty((0, 28), dtype=float, order='C')
for index, i in enumerate(df.sentence):
    position=df.position[index]
    tags = df.tags[index]
    question_title = df.question_title[index]
    explicit_features_for_df = np.append(explicit_features_for_df, np.array(explicit_features(i, position, tags, question_title)))
    explicit_features_for_df = explicit_features_for_df.reshape(-1, 28)

sentence_bert_for_df = np.empty((0, 768), dtype=float, order='C')

for i in trange(len(df.sentence)):
    i = df.sentence[i]
    encoded_input = tokenizer(i, return_tensors='pt')
    output = model(**encoded_input)
    sentence_bert_for_df = np.append(sentence_bert_for_df, output['pooler_output'].detach().numpy(), axis = 0)

return np.concatenate((sentence_bert_for_df, explicit_features_for_df), axis = 1)
  
new_howtosm = get_sm_from_sentence_classifier(howtomodel, test_set)
new_conceptualsm = get_sm_from_sentence_classifier(conceptualmodel, test_set)
new_debugsm = get_sm_from_sentence_classifier(debugmodel, test_set)

new_scores_from_sentence_classifier = np.concatenate((new_howtosm.reshape(1, -1), new_conceptualsm.reshape(1, -1), new_debugsm.reshape(1, -1)), axis = 0)

new_final_score = probs.T * new_scores_from_sentence_classifier

new_final_score = new_final_score[0] + new_final_score[1] + new_final_score[2]

print("With threshold 0.5, calculating model performance...")
new_truth = [1 if i > 0.1 else 0 for i in new_final_score]

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

precision = precision_score(dataset.truth.tolist(), new_truth)
recall = recall_score(dataset.truth.tolist(), new_truth)
f1 = 2 * precision * recall / (precision + recall)
print("precision:", precision)
print("recall:", recall)
print("F1:", f1)