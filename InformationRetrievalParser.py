import nltk
import urllib
import collections
from nltk import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem import PorterStemmer
from nltk import re
from nltk.corpus import stopwords
from itertools import groupby
from stemming.porter2 import stem
import math
from collections import Counter
from functools import reduce
import datetime

#Global variables/Lists to use in functions
count = 0
wordList = []
totalWords = 0
readCounts = []
list1 = []
list2 = []

#calculate tf
def tf(n1,n2):
    x = n1/n2
    return x

#calculate idf
def idf(x,y):
    val = x/y
    return val

#flatten function to flatten list (remove inner lists)
#https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-list-of-lists
def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def main():
    url1 = input("Enter the link of URL 1: ")
    url2 = input("Enter the link of URL 2: ")
    start = datetime.datetime.now()
    list1=(processURL(url1))
    list2=(processURL(url2))
    writeTotalCountOutput(wordList)
    calculateTfIdf(calculateTF(readCounts),calculateIDF(appearances(list1,list2)))
    finish = datetime.datetime.now()
    print ("Execution time:"+str(finish-start))
    


#processing of the url is done
def processURL(url):
    global count
    global wordList

    #html page parsing
    name = url
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html,features="html.parser")
    #remove javascript and css style from the parsed text.
    for js in soup(["script", "style"]):
        js.decompose()

    count+=1
    text = soup.get_text()

    
    #process parsed text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    #after each part of the processing, produces a file output for each document
    f = open('File '+ str(count) + ' HTML parsing output '+'.txt', 'w')
    f.write(text)
    print("File " + str(count) + " HTML Parsing output.txt saved")

    #Sentence splitting,tokenization and normalization process
    #tokenization
    tokens = word_tokenize(text)
    
    #remove punctuation
    tokens_nopunct = [word.lower() for word in tokens if re.search("\w",word)]
    f = open('File '+ str(count) + ' SS,Tokenization,Normalization output'+'.txt', 'w')
    f.write(str(tokens_nopunct))
    print("File " + str(count) + " SS,Tokenization,Normalization output.txt saved")

    #Stemming (Reduce a word to its word stem that affixes to suffixes and prefixes (roots))
    tokens_nopunct = [stem(word) for word in tokens_nopunct]
    f = open('File '+ str(count) + ' Stemming output'+'.txt', 'w')
    f.write(str(tokens_nopunct))
    print("File " + str(count) + " Stemming output.txt saved")

    #remove stopwords
    #https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens_nopunct if not w in stop_words]
    totalNumberOfWords = len(filtered)
    f = open('File '+ str(count) + ' Stopwords Removed'+'.txt', 'w')
    f.write(str(filtered))
    print("File " + str(count) + " Stopwords Removed.txt saved")

    
    
    #postagging
    tagged = nltk.pos_tag(filtered)
    length = len(tagged)
    f = open('File '+ str(count) + ' PosTagging output'+'.txt', 'w')
    f.write(str(tagged))
    wordList.append(filtered)
    print("File " + str(count) + " PosTagging output.txt saved")


    #count occurences of each word in document
    ##https://stackoverflow.com/questions/2600191/how-can-i-count-the-occurrences-of-a-list-item
    counts = [(i, len(list(c))) for i,c in groupby(sorted(flatten(tagged)))]
    f = open('File '+ str(count) + ' Count Output'+'.txt', 'w')
    f.write(str(counts))
    print("File " + str(count) + " Count Output.txt saved")
    
    return filtered


#Write the total count of each word from all documents to a file.
def writeTotalCountOutput(wList):
    global totalWords
    global readCounts
    allDocCounts = [(i, len(list(c))) for i,c in groupby(sorted(flatten(wList)))]
    f = open('AllCounts'+'.txt', 'w')
    f.write(str(allDocCounts))
    print("AllCounts.txt saved")
    totalWords = len(allDocCounts)
    readCounts.append(allDocCounts)


#Count how many lists a term appears in
#https://stackoverflow.com/questions/54807201/in-how-many-lists-does-a-term-appear
def appearances(list1,list2):
    all_lists = [list1, list2]
    newList = list(Counter(reduce(lambda x, y: x + list(set(y)), all_lists, [])).most_common())

    return newList
    

#TF calculation
def calculateTF(wList):
    #tf = occurences / totalwords
    global totalWords
    flattened = [val for sublist in wList for val in sublist]
    calculateTf = [(i,tf(c,totalWords)) for i,c in flattened]
 
    return calculateTf

#idf calculation
def calculateIDF(appList):
    #Calculate idf for each word: IDF(term) = log10(total # of documents / # of documents with term 
    calcIDF = [(i,math.log10(idf(count,c)))for i,c in appList]

    return calcIDF
    
    
#Calculate the Tf.idf by multiplying the above two calculations.
def calculateTfIdf(tfList,idfList):
    #https://stackoverflow.com/questions/54809374/taking-two-values-from-two-list-random-order-of-tuples-and-multiplying/54809455?noredirect=1#54809645
    ls1_new = sorted(tfList, key=lambda tup: tup[0])
    ls2_new = sorted(idfList, key=lambda tup: tup[0])


    val = [(t1, v1*v2) for (t1, v1), (t2, v2) in zip(ls1_new,ls2_new)]

    val.sort(key=lambda x: x[1], reverse = True)
    #https://stackoverflow.com/questions/8459231/sort-tuples-based-on-second-parameter

    #Write to a file that is formatted in rows and columns
    with open('IndexedTerms.txt', 'w') as f:
        f.write("Word "  + "Index_Score\n")
        for item in val:
            f.write("%s " % item[0])
            f.write("%s\n" % item[1])
    print("File saved to IndexedTerms.txt")
    

main()




