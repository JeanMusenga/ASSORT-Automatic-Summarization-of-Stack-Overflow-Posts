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
    # 找祈使句, 标记为0
    start = 0
    while start < len(taged) and taged[start][1] == taged[start][0]:
        start += 1
    if (start < len(taged) and taged[start][1] == "VB") or (start < len(taged) - 1 and taged[start][1].find("RB") != -1 and taged[start + 1][1] == -1):
        tags[0] = 1

    # 找情态动词（MD），标记为1
    start = 0
    while start < len(taged) - 1:
        if (taged[start][1].find("PRP") != -1 and taged[start + 1][1].find("MD") != -1):
            tags[1] = 1
        start += 1
    
    # 找JJR (比较级)， 标记为2
    if ("JJR" in allTags):
        tags[2] = 1

    # Key word: this work，标记为3
    if (k.lower().find("this work") != -1 and k.lower().find("code") != -1):
        tags[3] = 1

    # Key word: if，标记为4
    if k.lower().find("if") != -1:
        tags[4] = 1

    # Key word: to do，标记为5
    start = 0
    while start < len(taged) and taged[start][1] == taged[start][0]:
        start += 1
    if (start < len(taged) - 1 and taged[start][1].find("TO") != -1 and taged[start + 1][1].find("VB") != -1):
        tags[5] = 1

    # Key word: first，标记为6
    if k.lower().find("first") != -1:
        tags[6] = 1
    
    # Key word: then, second, 标记为7
    if k.lower().find("second") != -1 or k.lower().find("then") != -1:
        tags[7] = 1

    # Key word: third, final, 标记为8
    if k.lower().find("third") != -1 or k.lower().find("final") != -1:
        tags[8] = 1
    
    # Key word: the problem is, 标记为9
    if k.lower().find("the problem is") != -1:
        tags[9] = 1
    
    # Key word: rather than, 标记为10
    if k.lower().find("rather than") != -1:
        tags[10] = 1
    
    # Key word: solution is, 标记为11
    if k.lower().find("solution") != -1 and k.lower().find("is") != -1:
        tags[11] = 1

    # Key word: JJS, 标记为12
    if ("JJS" in allTags):
        tags[12] = 1

    # Key word: solve, 标记为13
    if k.lower().find("solve") != -1:
        tags[13] = 1

    # Key word: proper, 标记为14
    if k.lower().find("proper") != -1:
        tags[14] = 1

    # Key word: correct, 标记为15
    if k.lower().find("correct") != -1:
        tags[15] = 1

    # Key word: work, 标记为16
    if k.lower().find("work") != -1:
        tags[16] = 1

    # update, 标记为17
    if k.lower().find("update") != -1:
        tags[17] = 1

    # 是否和question用了同样的term？标记为18
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

# 当我们需要将所有embedding取平均值时，average为True，默认为False
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