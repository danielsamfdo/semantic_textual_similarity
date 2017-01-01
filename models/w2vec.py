import gensim
import lib.utilities as utility
import numpy as np
import math

def predict(sentence_1, sentence_2, w2vec_model):
  documents = [sentence_1, sentence_2]
  documents_tokens = utility.get_dict_vectors_of_documents(documents, justTokens=True)
  predicted_answers = []
  for i in range(len(documents_tokens)):
    s1_tokens, s2_tokens = documents_tokens[i]
    s1_vector = computation_vec_for_sentence(s1_tokens, w2vec_model)
    s2_vector = computation_vec_for_sentence(s2_tokens, w2vec_model)
    predicted_answers.append(5*cossim_dense(s1_vector, s2_vector))
  print predicted_answers[0], len(predicted_answers)
  # print "Error in Estimation of Word2Vec similarity: "+str(utility.evaluate(predicted_answers,answers))
  # print "Pearsons Correlation Measure of Word2Vec similarity: "+str(utility.evaluate_pearsons_coefficient(predicted_answers,answers))
  

def computation_vec_for_sentence(s1_tokens, model):
  vector = np.zeros((300,))
  for token in s1_tokens:
    if(token in model.vocab):
      vector += model[token]
  return vector

def length(vector):
  return np.power(np.sum(np.power(vector, 2)),0.5)

def cossim_dense(v1,v2):
    # v1 and v2 are numpy arrays
    # Compute the cosine simlarity between them.
    # Should return a number between -1 and 1
    l1 = length(v1)
    l2 = length(v2)
    print l1,l2
    if(l1==0):
      print "LENGTH IS ZERO L1"
      return 0
    elif(l2==0):
      print "LENGTH IS ZERO L2"
      return 0
    return math.exp(math.log(np.sum(np.multiply(v1,v2))) - math.log(l1) -math.log(l2))

def w2vec_model():
  print "LOADING WORD2VEC MODEL"
  model = gensim.models.Word2Vec.load_word2vec_format('/Users/danielsampetethiyagu/Downloads/GoogleNews-vectors-negative300.bin', binary=True)
  print "LOADED WORD2VEC MODEL"
  return model

def w2vec_for_pair_of_docs(documents, model):
  documents_tokens = utility.get_dict_vectors_of_documents(documents, justTokens=True)
  S1 = []
  S2 = []
  for i in range(len(documents_tokens)):
    s1_tokens, s2_tokens = documents_tokens[i]
    s1_vector = computation_vec_for_sentence(s1_tokens, model)
    s2_vector = computation_vec_for_sentence(s2_tokens, model)
    S1.append(s1_vector)
    S2.append(s2_vector)
  return S1, S2

def w2vec_similarity_measure_unsupervised(documents, answers):
  model = w2vec_model()
  documents_tokens = utility.get_dict_vectors_of_documents(documents, justTokens=True)
  predicted_answers = []
  for i in range(len(documents_tokens)):
    s1_tokens, s2_tokens = documents_tokens[i]
    s1_vector = computation_vec_for_sentence(s1_tokens, model)
    s2_vector = computation_vec_for_sentence(s2_tokens, model)
    predicted_answers.append(5*cossim_dense(s1_vector, s2_vector))
  print "Error in Estimation of Word2Vec similarity: "+str(utility.evaluate(predicted_answers,answers))
  print "Pearsons Correlation Measure of Word2Vec similarity: "+str(utility.evaluate_pearsons_coefficient(predicted_answers,answers))
  