# -*- coding: utf-8 -*-
"""
Created on Sun Aug 20 00:59:05 2017

@author: Amit
"""
import numpy as np
import pandas as pd
import pickle
import re    
from sklearn.feature_extraction.text import TfidfVectorizer


def tokenize(text):    
    text = re.sub("[^a-zA-Z]", " ", text)
    tokens = text.split(" ")
    return tokens

def classify():
    with open("x_result.pkl", 'rb') as f:
        tfidf = pickle.load(f)
        
    filename = 'model.sav'
    clf = pickle.load(open(filename, 'rb'))
    
    print ("Type exit() to exit")
    print ("Enter question: ")
    text = input()
    while (text != 'exit()'):
        
        response = tfidf.transform([text])
        pred_nd = (response.toarray())
        
        print (clf.predict(pred_nd)[0])
        print ("Enter question: ")
        text = input()

if __name__ == '__main__':
    classify()
