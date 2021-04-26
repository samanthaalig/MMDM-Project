# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 00:27:38 2021

@author: 13059
"""
import sys
import pandas as pd

#COVID-CT Reports
COVIDxl= pd.ExcelFile('COVID-CT-MetaInfo.xlsx')
COVIDxl.sheet_names

for sheet in COVIDxl.sheet_names:
    file= pd.read_excel(COVIDxl,sheet_name= sheet, usecols="H, K")
    file.to_csv(sheet + '.txt', header= True, index= False)
    
    
import collections
import matplotlib.pyplot as plt
# Read input file, note the encoding is specified here 
# It may be different in your text file
file = open('positive_captions.txt', encoding="utf8")
a= file.read()
# Stopwords
stopwords = set(line.strip() for line in open('stopwords.txt', encoding="utf8"))
stopwords = stopwords.union(set(['Figure', 'Patient', 'China', 'Wuhan', 'Beijing']))
# Instantiate a dictionary, and for every word in the file, 
# Add to the dictionary if it doesn't exist. If it does, increase the count.
wordcount = {}
# To eliminate duplicates, remember to split by punctuation, and use case demiliters.
for word in a.lower().split():
    word = word.replace(".","")
    word = word.replace(",","")
    word = word.replace(":","")
    word = word.replace("\"","")
    word = word.replace("!","")
    word = word.replace("â€œ","")
    word = word.replace("â€˜","")
    word = word.replace("*","")
    if word not in stopwords:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1
# Print most common word
n_print = int(input("How many most common words to print: "))
print("\nOK. The {} most common words are as follows\n".format(n_print))
word_counter = collections.Counter(wordcount)
for word, count in word_counter.most_common(n_print):
    print(word, ": ", count)
# Close the file
file.close()
# Create a data frame of the most common words 
# Draw a bar chart
lst = word_counter.most_common(n_print)
df = pd.DataFrame(lst, columns = ['Word', 'Count'])
df.plot.bar(x='Word',y='Count')

#Non-COVID-CT Reports55
nonCOVIDxl= pd.ExcelFile('COVID-CT-MetaInfo.xlsx')
nonCOVIDxl.sheet_names

for sheet in nonCOVIDxl.sheet_names:
    file= pd.read_excel(nonCOVIDxl,sheet_name= sheet, usecols="H, K")
    file.to_csv(sheet + '.txt', header= True, index= False)
    