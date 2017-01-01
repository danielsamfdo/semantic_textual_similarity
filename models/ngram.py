from __future__ import division
from collections import defaultdict
from collections import Counter
import lib.preprocess as process
import lib.utilities as utility
import math
import pdb
import numpy

# def ngram_vector_keys(tokens, ngram_size=1):
#   vector_keys = []
#   if(len(tokens)-(ngram_size-1)<=0):
#       return [tuple(tokens[0:])]
#   for i in range(len(tokens)-(ngram_size-1)):
#     vector_keys.append(tuple(tokens[i:i+(ngram_size)]))
#     # vector_keys.append(" ".join(tokens[i:i+(ngram_size)]))
#   return vector_keys

# def character_ngram_vector_keys(tokens, ngram_size=2):
#   vector_keys = []

#   for token in tokens:
#     if(len(token)-(ngram_size-1)<=0):
#       vector_keys.append(tuple(token[0:]))
#     else:
#       for i in range(len(token)-(ngram_size-1)):
#         vector_keys.append(tuple(token[i:i+(ngram_size)]))
#   return vector_keys

def ngram_keys(sent_1_tokens, sent_2_tokens, ngram_size, character_ngram):
  if character_ngram:
    set1 = set(utility.character_ngram_vector_keys(sent_1_tokens, ngram_size))
    set2 = set(utility.character_ngram_vector_keys(sent_2_tokens, ngram_size))
  else:
    set1 = set(utility.ngram_vector_keys(sent_1_tokens, ngram_size))
    set2 = set(utility.ngram_vector_keys(sent_2_tokens, ngram_size))
  return set1, set2

def ngram_weighted_value(keys, IDFScores, character_ngram=False):
  value = 0.0
  if not character_ngram:
    for key in keys:
      for i in range(len(key)):# This is a tuple
        if(type(key[i]) is not tuple):
          if key[i] in IDFScores.keys():
            value += IDFScores[key[i]]
        else:
          if key[i][0] in IDFScores.keys():
            value += IDFScores[key[i][0]] 
  else:
    for key in keys:
      if key in IDFScores.keys():
        value+=IDFScores[key]
  return value

def similarity_score(set1, set2, IDFScores=None, character_ngram=False, ngram_weighing=False):
  if not ngram_weighing:
    numerator = len(set1)
    denominator = len(set2)
  else:
    #print set1,character_ngram,IDFScores
    numerator = ngram_weighted_value(set1, IDFScores, character_ngram)
    denominator = ngram_weighted_value(set2, IDFScores, character_ngram)
    # INCASE IDF Values are not present for it
    if denominator == 0:
      numerator = len(set1)
      denominator = len(set2)
  try:
    value = numerator/float(denominator)
  except ZeroDivisionError:
    pdb.set_trace()
    value = float(len(set1))/len(set2)
  return value

def containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size=1, character_ngram=False, ngram_weighing=False, IDFScores=None):
  set1, set2 = ngram_keys(sent_1_tokens, sent_2_tokens, ngram_size, character_ngram)
  intersecting_keys = (set1.intersection(set2))
  containment_of_sentence_1_in_2 = similarity_score(intersecting_keys, set1, IDFScores, character_ngram, ngram_weighing)
  containment_of_sentence_2_in_1 = similarity_score(intersecting_keys, set2, IDFScores, character_ngram, ngram_weighing)
  # return containment_of_sentence_1_in_2, containment_of_sentence_2_in_1
  return numpy.mean([containment_of_sentence_1_in_2, containment_of_sentence_2_in_1])

def JaccardCoefficient(sent_1_tokens, sent_2_tokens, ngram_size=1, character_ngram=False, ngram_weighing=False, IDFScores=None):
  set1, set2 = ngram_keys(sent_1_tokens, sent_2_tokens, ngram_size, character_ngram)
  try:  
    value = similarity_score(set1.intersection(set2), set1.union(set2), IDFScores, character_ngram, ngram_weighing)
  except:
    pdb.set_trace()
  return value
  # return float(len(set1.intersection(set2)))/len(set1.union(set2))

def POSTags_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size=1, ngram_weighing=False, IDFScores=None):
  postag_tokens_1 = process.postags(sent_1_tokens)
  postag_tokens_2 = process.postags(sent_2_tokens)
  return JaccardCoefficient(postag_tokens_1, postag_tokens_2, ngram_size, False, ngram_weighing, IDFScores), containment_coefficienct(postag_tokens_1, postag_tokens_2, ngram_size, False, ngram_weighing, IDFScores)

def Lemma_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size=1, ngram_weighing=False, IDFScores=None):
  lemma_tokens_1 = process.lemmatize(sent_1_tokens)
  lemma_tokens_2 = process.lemmatize(sent_2_tokens)
  return JaccardCoefficient(lemma_tokens_1, lemma_tokens_2, ngram_size, False, ngram_weighing, IDFScores), containment_coefficienct(lemma_tokens_1, lemma_tokens_2, ngram_size, False, ngram_weighing, IDFScores)

def character_ngram_JaccardCoefficient_and_containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size=2, character_ngram=True, ngram_weighing=False, IDFScores=None):
  return JaccardCoefficient(sent_1_tokens, sent_2_tokens, ngram_size, character_ngram, ngram_weighing, IDFScores), containment_coefficienct(sent_1_tokens, sent_2_tokens, ngram_size, character_ngram, ngram_weighing, IDFScores)

def TFIDF(documents):
  Vocabulary = Counter()
  DocVectors = []
  IDFVector = Counter()
  No_of_Documents = float(len(documents))
  n = len(documents)
  for document in documents:
    print str(n) + " Documents remaining to process"
    n-=1
    tf_single_doc_count = Counter(process.tokens(document))
    Vocabulary+= tf_single_doc_count
    DocVectors.append(tf_single_doc_count)
    IDFVector += Counter(tf_single_doc_count.keys())
  # print IDFVector
  for key in IDFVector.keys():
    IDFVector[key] = math.log(No_of_Documents/(1+IDFVector[key]))
  # print IDFVector
  # IDFVector = defaultdict(lambda:0.0, dict((key,Vocabulary[key]*) for key in c.keys()))
  TFIDFScores = defaultdict(lambda:0.0, dict((key,Vocabulary[key]*IDFVector[key]) for key in Vocabulary.keys()))
  # print TFIDFScores, IDFVector, Vocabulary
  return TFIDFScores, Vocabulary, DocVectors, IDFVector

def CharacterIDFVector(documents, ngram_size=2):
  No_of_Documents = float(len(documents))
  n = len(documents)
  IDFVector = Counter()
  for document in documents:
    tokens = process.tokens(document)
    # print str(n) + " Documents remaining to process"
    n-=1
    IDFVector += Counter(set(utility.character_ngram_vector_keys(tokens, ngram_size)))
  #print IDFVector
  for key in IDFVector.keys():
    IDFVector[key] = math.log(No_of_Documents/(1+IDFVector[key]))
  return IDFVector

def DocvectorTFIDF(TFIDFScores, tokens):
  return defaultdict(lambda:0.0, dict((key,TFIDFScores[key] if key in TFIDFScores.keys() else 0.0) for key in tokens))

def cosinesimilarity(document1, document2, TFIDFScores):
  tokens1 = set(process.tokens(document1))
  tokens2 = set(process.tokens(document2))
  vector1 = DocvectorTFIDF(TFIDFScores, tokens1)
  vector2 = DocvectorTFIDF(TFIDFScores, tokens2)
  len_vector_1 = math.sqrt(sum({k: v**2 for k, v in vector1.items()}.values()))
  len_vector_2 = math.sqrt(sum({k: v**2 for k, v in vector2.items()}.values()))
  cosine_similarity_score = (utility.dict_dotprod(vector1,vector2))/float((len_vector_2*len_vector_1))
  return cosine_similarity_score

def cosinesimilarity_without_TFIDF(document1, document2):
  vector1 = Counter(process.tokens(document1))
  vector2 = Counter(process.tokens(document2))
  len_vector_1 = math.sqrt(sum({k: v**2 for k, v in vector1.items()}.values()))
  len_vector_2 = math.sqrt(sum({k: v**2 for k, v in vector2.items()}.values()))
  cosine_similarity_score = (utility.dict_dotprod(vector1,vector2))/float((len_vector_2*len_vector_1))
  return cosine_similarity_score
