import regex
# from test import *
from nltk import word_tokenize, sent_tokenize, pos_tag
from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np

embedding_model = SentenceTransformer('all-mpnet-base-v2')
MAGIC = 28

def findSubstring(k, sub):
    result = 0
    if k.find(sub) != -1:
        result = 1
        k = k.replace(sub, "")
        if sub != "<li>":
            if k.find(sub[0]+"/"+sub[1:]) != -1:
                k = k.replace(sub[0]+"/"+sub[1:], "")
    return result, k 

def cleanSentence(k):
    code = 0
    em = 0
    strong = 0
    li = 0
    code, k = findSubstring(k, "<code>")
    em, k = findSubstring(k, "<em>")
    strong, k = findSubstring(k, "<strong>")
    li, k = findSubstring(k, "<li>")

    k = k.replace("<code>", "")
    k = k.replace("<em>", "")
    k = k.replace("<strong>", "")
    k = k.replace("<li>", "")
    k = k.replace("BIGBLOCK", "example code")
    return code, em, strong, li, k

def localTag(k, questions, prev_sentence):
    code, em, strong, li, k = cleanSentence(k)
    taged = pos_tag(word_tokenize(k))
    allTags = [x[1] for x in taged]
    tags = [0] * MAGIC
    start = 0
    while start < len(taged) and taged[start][1] == taged[start][0]:
        start += 1
    if (start < len(taged) and taged[start][1] == "VB") or (start < len(taged) - 1 and taged[start][1].find("RB") != -1 and taged[start + 1][1] == -1):
        tags[0] = 1

    start = 0
    while start < len(taged) - 1:
        if (taged[start][1].find("PRP") != -1 and taged[start + 1][1].find("MD") != -1):
            tags[1] = 1
        start += 1
    
    if ("JJR" in allTags):
        tags[2] = 1

    if (k.lower().find("this work") != -1 and k.lower().find("code") != -1):
        tags[3] = 1

    if k.lower().find("if") != -1:
        tags[4] = 1

    start = 0
    while start < len(taged) and taged[start][1] == taged[start][0]:
        start += 1
    if (start < len(taged) - 1 and taged[start][1].find("TO") != -1 and taged[start + 1][1].find("VB") != -1):
        tags[5] = 1

    if k.lower().find("first") != -1:
        tags[6] = 1

    if k.lower().find("second") != -1 or k.lower().find("then") != -1:
        tags[7] = 1

    if k.lower().find("third") != -1 or k.lower().find("final") != -1:
        tags[8] = 1
    
    if k.lower().find("the problem is") != -1:
        tags[9] = 1
    
    if k.lower().find("rather than") != -1:
        tags[10] = 1
    
    if k.lower().find("solution") != -1 and k.lower().find("is") != -1:
        tags[11] = 1

    if ("JJS" in allTags):
        tags[12] = 1

    if k.lower().find("solve") != -1:
        tags[13] = 1

    if k.lower().find("proper") != -1:
        tags[14] = 1

    if k.lower().find("correct") != -1:
        tags[15] = 1

    if k.lower().find("work") != -1:
        tags[16] = 1

    if k.lower().find("update") != -1:
        tags[17] = 1

    questionTags = []
    for i in questions:
        questionTags += pos_tag(word_tokenize(i))
    
    for i in questionTags:
        if i[1] in ["NN", "NNP", "NNS", "NNPS"]:
            k.lower().find(i[0]) != -1
            tags[18] = 1
            break
    
    tags[19] = code
    tags[20] = em
    tags[21] = strong
    tags[22] = li

    # big block
    if k.find("BIGBLOCK") != -1:
        tags[23] = 1

    # :
    if prev_sentence.strip() != '' and prev_sentence.strip()[-1] == ":":
        tags[24] = 1

    if k.lower().find("alterna") != -1:
        tags[25] = 1

    if k.lower().find("flaw") != -1:
        tags[26] = 1

    lenSen = min(20, len(taged))
    tags[27] = float(lenSen) / 20
    return tags, k

def cosineSimilarity(a, b):
    return 1 - spatial.distance.cosine(a, b)

def calInput(question_embedding, question_sentences, answer_sentences):
    cleaned_answer_sentences = []
    localTags = []
    for index, i in enumerate(answer_sentences):
        if index == 0:
            prev = ''
        else:
            prev = answer_sentences[index - 1]
        tag_vector, cleaned_sentence = localTag(i, question_sentences, prev)
        # input(question_sentences)
        cleaned_answer_sentences.append(cleaned_sentence)
        localTags.append(tag_vector)
    
    answer_embeddings = getEmbedding(cleaned_answer_sentences)
    
    return cleaned_answer_sentences, combine(localTags, question_embedding, answer_embeddings)

def getEmbedding(sentences, average=False):
    global embedding_model
    result = [embedding_model.encode([i])[0] for i in sentences]
    if average:
        return np.mean(np.array(result), axis = 0)
    return result

def combine(local_tags, question_embedding, answer_embeddings):
    result = []
    for index, _ in enumerate(local_tags):
        if index == 0:
            start = [1]
        else:
            start = [0]
        each_local = local_tags[index]
        each_answer_embedding = answer_embeddings[index]
        gold = start + each_local + list(each_answer_embedding) + [cosineSimilarity(question_embedding, each_answer_embedding)] + [len(local_tags)]
        result.append(gold)
    return result