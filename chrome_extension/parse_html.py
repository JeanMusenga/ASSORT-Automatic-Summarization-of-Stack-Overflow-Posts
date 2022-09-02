from nltk.data import OpenOnDemandZipFile
import regex
# from test import *
from nltk import word_tokenize, sent_tokenize, pos_tag
# from sentence_transformers import SentenceTransformer
from scipy import spatial
import numpy as np
from calInput import *
from predict import *

'''
Constants below:
'''
lineBreakers = ["/p", "br", "/h3", "/h2", "/h3", "/h4", "/h1", "/h5", "/li"]
preserve = ["code", "/code", "em", "/em", "strong", "/strong", "li"]

'''
Helper functions
'''
def getIndexList(data):
    # Get rid of the block codes first
    # data.replace("\n", "")
    a = regex.compile('<pre.*?>.*?</pre>', regex.DOTALL)
    codes = []
    code_start_indexes = []
    actual_codes = []
    prematches = list(regex.finditer(a, data))
    karma = 0
    for xindex, x in enumerate(prematches):
        code_start_indexes.append(x.span()[0] - karma)
        karma += (x.span()[1] - x.span()[0])
        actual_codes.append(x.group())
        codes.append((x.span()[0], x.span()[1]))

    

    short = ""
    prev = 0
    for i in codes:
        short += data[prev:i[0]]
        prev = i[1]
    short += data[prev:]

    b = regex.compile('((?<=<[^/].*?>)((?!</.+?>)[^<])+?(?=<[^/].*?>))|((?<=<[^/].*?>)((?!<[^/].*?>)[^<])+?(?=</.*?>))|((?<=</.*?>)((?!</.+?>)[^<])+?(?=<[^/].*?>))|((?<=</.*?>)((?!<[^/].+?>)[^<])+?(?=</.*?>))', regex.DOTALL)
    text = [x for x in regex.finditer(b, short)]

    # text = [x.group() for x in regex.finditer( r'((?<=</.+?>).+(?=<.+?>))|((?<=<.+?>).+(?=<.+?>))|((?<=</.+?>).+(?=</.+?>))|((?<=<.+?>).+?(?=</.+?>)(?!=<.+?>))', data)]
    index_list = []
    stream_of_char = ''
    to_sentence = ''
    for i in text:
        start = i.span()[0]
        end = i.span()[1]
        for j in range(start, end):
            if short[j] not in [" ", "\n"]:
                index_list.append(j)
        to_sentence += i.group().replace("\n", '')
        stream_of_char += i.group().replace(" ", "").replace("\n", '')
    cumulate = ''
    for i in index_list:
        cumulate += short[i]
    return short, code_start_indexes, actual_codes, stream_of_char.lower(), index_list, to_sentence

def contains(a, linebreakers):
    return any([a.find(x) != -1 for x in linebreakers])

# 当我们不需要考虑块状代码时，flag设为False，default为True
def clean(data, flag=True):
    # Remove pre block first
    a = regex.compile("<pre.*?>.*?</pre>", regex.DOTALL)
    pres = [x for x in regex.finditer(a, data)]
    temp = ''
    prev = 0
    for i in pres:
        # if flag:
        #     temp += (data[prev:i.span()[0]] + "\nBIGBLOCK\n")
        # else: 
        #     temp += data[prev:i.span()[0]]
        temp += data[prev:i.span()[0]]
        temp += "<bonanblock>"
        prev = i.span()[1]
    temp += data[prev:]
    temp = temp.replace("\n", "")
    # Remove other tags
    result = ''
    tags = [x for x in regex.finditer(r'<.+?>', temp)]
    prev = 0
    for i in tags:
        if i.group().find("<bonanblock>") != -1:
            result += (temp[prev:i.span()[0]] + "\nBIGBLOCK\n")
        elif contains(i.group(), lineBreakers):
            result += temp[prev:i.span()[0]] + "\n"
        elif any([i.group()[1:].startswith(x) for x in preserve]):
            if flag:
                result += temp[prev:i.span()[1]]
            else:
                result += temp[prev:i.span()[0]]
        else:
            result += temp[prev:i.span()[0]]
        prev = i.span()[1]
    result += temp[prev:]

    sentences = []
    codeIndexes = []
    for i in result.split("\n"):
        if i != "" and i != "BIGBLOCK":
            for j in sent_tokenize(i):
                if j != "":
                    sentences.append(j)
        elif i == "BIGBLOCK":
            codeIndexes.append(len(sentences) - 1)
    return sentences, codeIndexes

def insert(operate_on, index_list, insertion, position, codes):
    length = len(insertion)
    operate_on = operate_on[:position] + insertion + operate_on[position:]
    for index, i in enumerate(index_list):
        if i >= position:
            index_list[index] += length
    for index, i in enumerate(codes):
        if i >= position:
            codes[index] += length
    return operate_on, index_list, codes

def produceCSS(classList, sig):
    result = '<span class="'
    for i in classList:
        result += '{} '.format(i)
    result = result[:-1]
    result += '"'
    result += ' style="--acolor:{};"'.format(max((sig - 0.2) / 0.8, 0))
    result += '>'
    return result

def addTag(operate_on, index_list, stream_of_char, normalized_sentence, index, id, selected, sig, codes, where_are_we):
    if len(stream_of_char) != 0:
        left_index = index_list[stream_of_char[where_are_we:].find(normalized_sentence) + where_are_we]
        new_where_are_we = stream_of_char[where_are_we:].find(normalized_sentence) + len(normalized_sentence) + where_are_we
        right_index = index_list[stream_of_char[where_are_we:].find(normalized_sentence) + where_are_we + len(normalized_sentence) - 1] + 1
        b = regex.compile('(((?<=^\s*?)[^\s]((?!<.+?>).)*?(?=\s*$))|((?<=^\s*?)[^\s]((?!<.+?>).)+?(?=<[^/].*?>))|((?<=^\s*?)[^\s]((?!<.+?>).)+?(?=</.*?>))|((?<=<[^/].+?>)((?!<.+?>).)+?(?=\s*$))|((?<=</.+?>)((?!<.+?>).)+?(?=\s*$))|(?<=<[^/].*?>)((?!</.+?>).)+?(?=<[^/].*?>))|((?<=<[^/].*?>)((?!<[^/].*?>).)+?(?=</.*?>))|((?<=</.*?>)((?!</.+?>).)+?(?=<[^/].*?>))|((?<=</.*?>)((?!<[^/].+?>).)+?(?=</.*?>))', regex.DOTALL)
        text = [x for x in regex.finditer(b, operate_on[left_index:right_index])]

        accumulate = 0
        for i in text:
            classList = []
            classList.append("so_tracker_sentence")
            classList.append("index_{}".format(index))
            classList.append("belong_{}".format(id))
            if selected:
                classList.append("extracted")
            if i.span()[0] == 0:
                classList.append("leftmost")
            if i.span()[1] == right_index-left_index:
                classList.append("rightmost")
            left = produceCSS(classList, sig)
            right = "</span>"
            operate_on, index_list, codes = insert(operate_on, index_list, left, left_index + accumulate + i.span()[0], codes)
            accumulate += len(left)
            operate_on, index_list, codes = insert(operate_on, index_list, right, left_index + accumulate + i.span()[1], codes)
            accumulate += len(right)
    else:
        new_where_are_we = where_are_we
    return operate_on, index_list, codes, new_where_are_we

def markPre(i, sig):
    style = 'style= "--acolor:{};"'.format(max((sig - 0.2) / 0.8, 0))
    i = (i.split(" ", 1)[0] + " " + style + " " + i.split(" ", 1)[1])
    # i = (i.split('class="', 1)[0] + 'class="' + "selected_pre"+ " " + i.split('class="', 1)[1])
    i = (i.split('class="', 1)[0] + 'class="' + ""+ " " + i.split('class="', 1)[1])
    return i

def putCodeBack(operate_on, codes, actual_codes, codeIndexes, pred_index, pred_sig):
    for index, i in enumerate(actual_codes):
        if codeIndexes[index] in pred_index:
            sig = pred_sig[pred_index.index(codeIndexes[index])]
            try:
                i = markPre(i, sig)
            except:
                print("乘风破浪会有时", i)
        cindex = codes[index]
        operate_on, _, codes = insert(operate_on, [], i, cindex, codes)
    return operate_on

def normalizeSentence(i):
    i = regex.sub(r"<.+?>", "", i)
    return i.replace(" ", '').lower()

def parse(question, answers, answerIds):
    question_sentences, _ = clean(question, False)
    question_embedding = getEmbedding(question_sentences, True)
    ultimate = []
    best_predictions = []
    code_exist = []

    # 也许一次只输入一个answer较好？
    for answer_index, answer in enumerate(answers):
        where_are_we = 0
        id = answerIds[answer_index]
        operate_on, codes, actual_codes, stream_of_char, index_list, to_sentence = getIndexList(answer)
        answer_sentences, codeIndexes = clean(answer)
        answer_sentences, model_inputs = calInput(question_embedding, question_sentences, answer_sentences)
        print("此去经年", answer_sentences[0])
        # Initialize model & make prediction
        prediction, best = predict(model_inputs, 0.90)
        best_predictions.append(answer_sentences[best])
        # Modify operate_on accordingly
        for index, i in enumerate(answer_sentences):
            pred_index = [x[0] for x in prediction]
            pred_sig = [x[1] for x in prediction]
            selected = (index in pred_index)
            if selected:
                sig = pred_sig[pred_index.index(index)]
            else:
                sig = 0
            
            operate_on, index_list, codesm, where_are_we = addTag(operate_on, index_list, stream_of_char, normalizeSentence(i), index, id, selected, sig, codes, where_are_we)
        operate_on = putCodeBack(operate_on, codes, actual_codes, codeIndexes, pred_index, pred_sig)
        ultimate.append(operate_on)
        if codes != []:
            code_exist.append("yes")
        else:
            code_exist.append("no")

    return ultimate, best_predictions, code_exist


if __name__ == "__main__":
    data1 = '''b'\n<p>Let\'s use <a href="https://pandas.pydata.org/docs/reference/api/pandas.Series.shift.html" rel="nofollow noreferrer"><code>shift</code></a> instead to "shift" the column up so that rows are aligned with the previous, then use <a href="https://pandas.pydata.org/docs/reference/api/pandas.Series.lt.html" rel="nofollow noreferrer"><code>lt</code></a> to compare less than and <a href="https://pandas.pydata.org/docs/reference/api/pandas.Series.astype.html" rel="nofollow noreferrer"><code>astype</code></a> convert the booleans to 1/0:</p>\n<pre class="lang-py s-code-block"><code>df[\'out\'] = df[\'col1\'].lt(df[\'col1\'].shift(-1)).astype(int)\n</code></pre>\n<pre class="lang-py s-code-block"><code>   col1  out\n0     1    1\n1     3    0\n2     3    0\n3     1    1\n4     2    1\n5     3    0\n6     2    0\n7     2    0\n</code></pre>\n'''
    data3 = '''b'\n<p>In addition to the differences already noted, there\'s another extremely important difference that I just now discovered the hard way:  unlike <code>np.mean</code>, <code>np.average</code> doesn\'t allow the <code>dtype</code> keyword, which is essential for getting correct results in some cases.  I have a very large single-precision array that is accessed from an <code>h5</code> file.  If I take the mean along axes 0 and 1, I get wildly incorrect results unless I specify <code>dtype=\'float64\'</code>:</p>\n\n<pre class="lang-py s-code-block"><code class="hljs language-python">&gt;T.shape\n(<span class="hljs-number">4096</span>, <span class="hljs-number">4096</span>, <span class="hljs-number">720</span>)\n&gt;T.dtype\ndtype(<span class="hljs-string">\'&lt;f4\'</span>)\n\nm1 = np.average(T, axis=(<span class="hljs-number">0</span>,<span class="hljs-number">1</span>))                <span class="hljs-comment">#  garbage</span>\nm2 = np.mean(T, axis=(<span class="hljs-number">0</span>,<span class="hljs-number">1</span>))                   <span class="hljs-comment">#  the same garbage</span>\nm3 = np.mean(T, axis=(<span class="hljs-number">0</span>,<span class="hljs-number">1</span>), dtype=<span class="hljs-string">\'float64\'</span>)  <span class="hljs-comment"># correct results</span>\n</code></pre>\n\n<p>Unfortunately, unless you know what to look for, you can\'t necessarily tell your results are wrong.  I will never use <code>np.average</code> again for this reason but will always use <code>np.mean(.., dtype=\'float64\')</code> on any large array.  If I want a weighted average, I\'ll compute it explicitly using the product of the weight vector and the target array and then either <code>np.sum</code> or <code>np.mean</code>, as appropriate (with appropriate precision as well).</p>\n'''
    
    data = ['''b'\n<pre>dfjaldkfja;d</pre>''']

    question1 = "What is a non-capturing group in regular expressions? How are non-capturing groups, i.e. (?:), used in regular expressions and what are they good for?"
    question2 = "Is it possible to write single line return statement with if statement?Is is possible to return from a method in single line in python? Looking for something like this. Tried above, and it is invalid syntax. I could easily do. But just curious if I can combine above if statement into a single line."
    question3 = "np.mean() vs np.average() in Python NumPy?I notice that.However, there should be some differences, since after all they are two different functions.What are the differences between them?"
    parse(question = question1, answers = data, answerIds=[1, 2, 3, 4, 5, 6])
