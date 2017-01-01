#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 14:19:32 2016

@author: sanketh
"""
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import CorpusReader as cr
import re
from collections import defaultdict
from nltk.corpus import brown
from nltk.probability import FreqDist
from nltk.corpus import gutenberg
import math
import cProfile
from collections import Counter
import pickle



print(wn.synset('dog.n.01').definition())

def remove_punctuation(sentence11):
    sentence1 = re.sub("[,.?]", "", sentence11)
    sent = sentence1.lower()
    return sent


def list_creation(sentence1, sentence2):
    list = []
    for i in sentence1.split():
        if i not in list:
            list.append(i)
    for j in sentence2.split():
        if j not in list:
            list.append(j)
    str1 = ' '.join(list)
    print list
    return str1

def semantic_similarity(combined_sent, curr_sentence, tot_count, brown_dict):
    dict1 = defaultdict(float)
    for each_word in combined_sent.split():
        s_tilda = 0
        for each in curr_sentence.split():
          if each_word==each:
              first_word= each
              second_word = each_word
              s_tilda = 1
          else:
            
            first = wn.synsets(each)
            second = wn.synsets(each_word)
            sim = 0
            if len(first)>0 and len(second)>0:
                sim =  first[0].path_similarity(second[0])
            if sim > s_tilda and sim>0.2:
                s_tilda = sim
                first_word = each
                second_word = each_word
        print "sdasd", each_word, s_tilda
        first_word_count = 0
        second_word_count = 0
        print "DEBUGGING INFO"
        # print first_word_count,second_word_count
        # x = brown.words() 
        # print len(x), len(set(x))
        if s_tilda!=0:
            first_word_count = brown_dict[first_word]
            second_word_count = brown_dict[second_word]
        # for m1 in brown.words():
        #  if s_tilda!=0:
        #     if m1 == first_word:
        #         first_word_count+=1
        #     if m1 == second_word:
        #         second_word_count+=1
        first_info = 0
        second_info = 0
        first_info = 1-(math.log(first_word_count+1)/math.log(tot_count+1))
        second_info = 1-(math.log(second_word_count+1)/math.log(tot_count+1))
        dict1[each_word] = first_info*second_info*s_tilda
    return dict1
def semantic_calc(dict1, dict2):
    dot_prod=0
    for x, y in dict2.items():
        dot_prod+=(dict1[x]*dict2[x])   
    dict_fir = 0
    for x, y in dict1.items():
        dict_fir+=y*y
    fin_1 = math.sqrt(dict_fir)
    dict_sec = 0
    for x, y in dict2.items():
        dict_sec+=y*y
    fin_2 = math.sqrt(dict_sec)
    final_ans = dot_prod/(fin_1*fin_2)
    return final_ans




#word order


        
        
tot_count = 0

def brown_count():
    tot_count = 0
    if tot_count > 0:
        return tot_count
    tot_count = len(brown.words()) 
    return tot_count

def brown_dict():
    words = brown.words()
    lower_cased_words = map(lambda wrd:wrd.lower(), words)
    brown_dict = Counter(lower_cased_words)
    data = {}
    data.update({"brown_count": brown_count(), "brown_dict": brown_dict})
    pickle.dump( data, open( "brown_dict.pkl", "wb" ) )
    return brown_dict

def brown_dict():
    words = brown.words()
    lower_cased_words = map(lambda wrd:wrd.lower(), words)
    brown_dict = Counter(lower_cased_words)
    pickle.dump( brown_dict, open( "brown_dict.pkl", "wb" ) )
    return brown_dict

if __name__ == "__main__":

    d = pickle.load(open("brown_dict.pkl","r"))
    tot_count = d["brown_count"]
    brown_dict = d["brown_dict"]
    sent11 = "The black lion jumping under the tiger"
    sent22 = "A quick brown fox jumps over the lazy dog"
    final_sentence1 = remove_punctuation(sent11)
    print final_sentence1
    final_sentence2 = remove_punctuation(sent22)
    print final_sentence2
    sentence_string = list_creation(final_sentence1, final_sentence2)
    print sentence_string
    combined_sentence = remove_punctuation(sentence_string)
    tot_count = brown_count()

    dict1 = semantic_similarity(combined_sentence, final_sentence1, tot_count, brown_dict)
    dict2 = semantic_similarity(combined_sentence, final_sentence2, tot_count, brown_dict)
    print "dictionary1", dict1
    print "dictionary2", dict2    
    sem_similarity = semantic_calc(dict1, dict2)
    print "semsem", sem_similarity
    
    


    
    
    
    
























#dog = wn.synsets('things')
#boxx = wn.synsets('cpu')
#print boxx
##print boxx.hypernyms()
##print dog.name()
##for i in range(0,len(dog)):
##    print "dog", dog[i]
##    print dog[i].hypernyms()
##print wn.synsets('program')
#synonyms = []
#antonyms = []
#for syn in wn.synsets('good'):
#    for l in syn.lemmas():
#        synonyms.append(l.name())
#        if l.antonyms():
#            antonyms.append(l.antonyms()[0].name())
#            
#print (set(synonyms))
#print (set(antonyms))
#if len(dog)>0 and len(boxx)>0:
#    print dog[0].path_similarity(boxx[0])
#    print dog[0].lch_similarity(boxx[0])
#    print dog[0].wup_similarity(boxx[0])
