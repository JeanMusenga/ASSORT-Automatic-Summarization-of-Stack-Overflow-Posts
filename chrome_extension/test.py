import nltk
from nltk import word_tokenize, sent_tokenize, pos_tag
import sys
import re
import pickle
import os
import copy
import pyodbc
from os import listdir
from os.path import isfile, join

lineBreakers = ["/pre", "/p", "br", "/h3", "/h2", "/h3", "/h4", "/h1", "/h5", "/li"]

def notInCode(substringStartIndex, substringEndIndex, string):
    beforeStartIndex = string.rfind("<code>", 0, substringStartIndex)
    beforeEndIndex = string.rfind("</code>", 0, substringStartIndex)
    afterStartIndex = string.find("<code>", substringEndIndex, len(string))
    afterEndIndex = string.find("</code>", substringEndIndex, len(string))
    if beforeStartIndex != -1 and afterEndIndex != -1:
        if beforeStartIndex < beforeEndIndex:
            return True
        elif afterStartIndex > afterEndIndex:
            return True
        else:
            return False
    else:
        return True
    return not (beforeStartIndex > beforeEndIndex and afterEndIndex != -1 and ((afterEndIndex < afterStartIndex and afterStartIndex != -1) or afterStartIndex == -1))

def clean(string, flag):
    string = string.replace("\n", "")
    tags = []
    with open("tags.txt", "r") as f:
        tag = f.readline()
        tags.append(tag)
        while tag != "":
            tag = f.readline()
            tags.append(tag)
        tags = tags[:-1]
    for i in tags:
        temp = []
        if len(i.split(" ")) == 1:
            first = i.split(" ")[0]
            i = i.split(" ")[0][1:len(first)-2].lower()
        else:
            i = i.split(" ")[0][1:].lower()
        origin = i
        patternA = re.compile('<%s .*?>'%i)
        patternB = re.compile('<%s>'%i)
        occurancesA = [(m.start(0), m.end(0)) for m in re.finditer(patternA, string)]
        occurancesB = [(m.start(0), m.end(0)) for m in re.finditer(patternB, string)]
        occurances = list(set().union(occurancesA, occurancesB))
        
        for occurance in occurances:
            substringStartIndex = occurance[0]
            substringEndIndex = occurance[1]
            if notInCode(substringStartIndex, substringEndIndex, string):
                if flag == 2:
                    if i in lineBreakers and not i == "/pre":
                        temp.append((substringStartIndex, substringEndIndex, 0))

                    elif i == "/pre":
                        temp.append((substringStartIndex, substringEndIndex, 2))
                    elif i not in ["code", "/code", "em", "/em", "strong", "/strong", "li"]:
                        temp.append((substringStartIndex, substringEndIndex, 1))
                else:
                    if i in lineBreakers:
                        temp.append((substringStartIndex, substringEndIndex, 0))
                    elif i not in ["code", "/code", "em", "/em", "strong", "/strong", "li"]:
                        temp.append((substringStartIndex, substringEndIndex, 1))
        temp.sort(key=lambda y: y[0])
        prev = 0

        if len(temp) != 0:
            result = ''
            for i in temp:
                if i[2] == 1:
                    result += string[prev:i[0]]
                elif i[2] == 2:
                    result += (string[prev:i[0]] + "\nBIGBLOCK\n")
                else:
                    result += (string[prev:i[0]] + "\n")
                prev = i[1]
            result += string[prev:len(string)]
            string = result
    tempList = []
    for i in string.split("\n"):
        if i != "":
            tempList.append(i)
    finalResult = ''
    for index, i in enumerate(tempList):
        finalResult += i
        if index != len(tempList) - 1:
            finalResult += '\n'
    return finalResult

def initializeDatabase():
    # Connect to a database
    database = "StackOverflow2010"
    connect_string = 'Driver={SQL Server}; Server=.\SQLEXPRESS; Database=StackOverflow2010; Trusted_Connection=yes;'

    # Cursor for question
    conn = pyodbc.connect(connect_string)
    cursor = conn.cursor()

    # Cursor for answer
    conn2 = pyodbc.connect(connect_string)
    cursor2 = conn2.cursor()

    print("Fetching data")
    cursor.execute("select * from Posts where PostTypeId = 1 order by ViewCount desc")
    return cursor, cursor2

idlist = []
def categorize(id, body, title, index, cursor2, count):
    toWrite = ""
    body = clean(body, 1)
    title = clean(title, 1)

    # Let's ask Bonan what type of question it is!
    print("第", index, "({})题目：".format(id), title)
    print("第", index, "题干：", body)

    category = input("请博楠判断此题为如下的，（1）What-is, （2）How-to, （3）Debug,（8）输入新LB，ENTER跳过: ")

    while category not in ['1', '2', '3', '8', '']:
        print("格式错误！")
        print("题目：", title)
        print("题干：", body)
        category = input("请博楠重新判断此题为如下的，（1）What-is, （2）How-to, （3）Debug，（8）输入新LB，ENTER跳过: ")

    if category in ['1', '2', '3']:
        count[int(category) - 1] += 1
        # Now the result from Bonan 
        file = "oct/{}_{}.txt".format(category, id)
        print("将储存至", file, "。")

        # At first, we want add title and body of question to the output string
        toWrite += title
        toWrite += "\n"
        toWrite += body
        toWrite += "MYSPECIALBREAK"
        cursor2.execute("select * from Posts where PostTypeId = 2 and ParentId = {}".format(id))

        for answer in cursor2:
            answerBody = clean(answer[3], 2)
            toWrite += answerBody
            # Line breaker
            toWrite += "MYSPECIALBREAK"
        
        with open(file, 'w', encoding="utf-8") as f:
            f.write(toWrite)
        pickle.dump(count, open("count.txt", "wb"))
    
        print("目前我们已经有，{}个What-is，{}个How-to，{}个Debug。".format(count[0], count[1], count[2]))
        print("继续革命。\n\n")
    elif category == "":
        print("目前我们仍有，{}个What-is，{}个How-to，{}个Debug。".format(count[0], count[1], count[2]))
        print("此题跳过。\n\n")
    else:
        newLineBreak = input("新line break！")
        if newLineBreak != "":  
            lineBreakers.append(newLineBreak)
        categorize(id, body, title, index, cursor2)

def dontHave(id):
    for i in listdir("fake") + listdir("oct"):
        if i.find(str(id)) != -1:
            return False
    return True

def loop(cursor, cursor2, count):
    for index, row in enumerate(cursor):
        if index > 99999:
            break
        if index > 5000:
            id = row[0]
            title = row[18]
            body = row[3]

            # if dontHave(id):
            #     categorize(id, body, title, index, cursor2, count)
            if dontHave(id) and (body.find("exception") != -1 or body.find(" bug") != -1):
                categorize(id, body, title, index, cursor2, count)

def updateCount():
    count = [0, 0, 0]
    # files = listdir("fake") + listdir("oct")
    files = listdir("oct")
    for i in files:
        if i.find("1_") != -1:
            count[0] += 1
        elif i.find("2_") != -1:
            count[1] += 1
        else:
            count[2] += 1
    return count

def main():
    count = updateCount()
    cursor, cursor2 = initializeDatabase()
    loop(cursor, cursor2, count)
    # for i in cursor:
    #     print(clean(i[18]))
    #     print(clean(i[3]))
    #     input("\n\nNext")

if __name__ == "__main__":
    main()



































# for index, row in enumerate(cursor):
#     if index > 99:
#         break
#     print(clean(row[3]))
#     truth = input("Press any key to proceed")

# sys.exit()

# test = ""
# index = test.find("ad")
# print(test[:index] + test[index + 2:])


# sys.exit()
# test = """When you declare a reference variable (i.e., an object), you are really creating a pointer to an object. Consider the following code where you declare a variable of primitive type int:
# In this example, the variable x is an int and Java will initialize it to 0 for you. When you assign the value of 10 on the second line, your value of 10 is written into the memory location referred to by x.
# But, when you try to declare a reference type, something different happens. Take the following code:
# The first line declares a variable named num, but it does not actually contain a primitive value yet. Instead, it contains a pointer (because the type is Integer which is a reference type). Since you have not yet said what to point to, Java sets it to null, which means <strong>"I am pointing to <em>nothing<em>".</strong>"""

# test = test.split("\n")
# result = []
# for i in test:
#     eachParagraph = sent_tokenize(i)
#     for j in eachParagraph:
#         result.append(j)

# print(result)