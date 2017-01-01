import pickle
import dill
import lib.preprocess as process
from collections import Counter
import numpy as np
# import models.ngram as ng

def ngram_vector_keys(tokens, ngram_size=1):
  vector_keys = []
  if(len(tokens)-(ngram_size-1)<=0):
      return [tuple(tokens[0:])]
  for i in range(len(tokens)-(ngram_size-1)):
    vector_keys.append(tuple(tokens[i:i+(ngram_size)]))
    # vector_keys.append(" ".join(tokens[i:i+(ngram_size)]))
  return vector_keys

def character_ngram_vector_keys(tokens, ngram_size=2):
  vector_keys = []

  for token in tokens:
    if(len(token)-(ngram_size-1)<=0):
      vector_keys.append(tuple(token[0:]))
    else:
      for i in range(len(token)-(ngram_size-1)):
        vector_keys.append(tuple(token[i:i+(ngram_size)]))
  return vector_keys

def dict_dotprod(d1, d2):
  """Return the dot product (aka inner product) of two vectors, where each is
  represented as a dictionary of {index: weight} pairs, where indexes are any
  keys, potentially strings.  If a key does not exist in a dictionary, its
  value is assumed to be zero."""
  smaller = d1 if len(d1)<len(d2) else d2  # BUGFIXED 20151012
  total = 0
  for key in smaller.iterkeys():
      total += d1.get(key,0) * d2.get(key,0)
  return total

def save_weights(path,weights):
  prefix ="weights/"
  pickle.dump(weights, open(prefix+path, 'wb'))

def load_weights(path):
  return pickle.load(open(path, 'rb'))

def evaluate_pearsons_coefficient(gold_standard, predicted_answers):
  N = len(gold_standard)
  gs = np.array(gold_standard)
  ans = np.array(predicted_answers)
  gs_pow_2 = np.power(gs, 2)
  # print gs_pow_2
  ans_pow_2 = np.power(ans,2)
  # print ans_pow_2
  gs_ans_mul = np.multiply(gs,ans)
  numerator = ((N*np.sum(gs_ans_mul)) - (np.sum(gs)*np.sum(ans)))
  denominator = np.power(( (N*np.sum(gs_pow_2)) - np.power(np.sum(gs),2) ) * ( (N*np.sum(ans_pow_2)) - np.power(np.sum(ans),2) ), 0.5)
  return numerator/denominator

def evaluate(gold_standard, predicted_answers):
  return np.sum(np.absolute(np.array(gold_standard) - np.array(predicted_answers)))/len(gold_standard) 
  # return math.sqrt(np.sum(np.power(np.array(gold_standard) - np.array(predicted_answers),2))) 

def get_dict_vector_of_2_sentences(sentence_1_tokens, sentence_2_tokens):
  # print sentence_1_tokens,sentence_2_tokens
  # print dict(Counter(sentence_1_tokens)+Counter(sentence_2_tokens))
  token_counter = Counter()
  pos_counter = Counter()
  lemma_counter = Counter()
  char_counter = Counter()
  # print sentence_1_tokens,sentence_2_tokens
  pos_tokens_1 = process.lemmatize(sentence_1_tokens)
  pos_tokens_2 = process.lemmatize(sentence_2_tokens)
  lemma_tokens_1 = process.postags(sentence_1_tokens)
  lemma_tokens_2 = process.postags(sentence_2_tokens)
  for n in range(1,5):
    token_counter += Counter(ngram_vector_keys(sentence_1_tokens, ngram_size=n))+Counter(ngram_vector_keys(sentence_2_tokens, ngram_size=n))
    pos_counter += Counter(ngram_vector_keys(pos_tokens_1, ngram_size=n))+Counter(ngram_vector_keys(pos_tokens_2, ngram_size=n))
    lemma_counter += Counter(ngram_vector_keys(lemma_tokens_1, ngram_size=n))+Counter(ngram_vector_keys(lemma_tokens_2, ngram_size=n))
    if(n>=2):
      char_counter += Counter(character_ngram_vector_keys(sentence_1_tokens, ngram_size=n))+Counter(character_ngram_vector_keys(sentence_2_tokens, ngram_size=n))
  # print dict(token_counter+pos_counter+lemma_counter+char_counter)
  return dict(token_counter+pos_counter+lemma_counter+char_counter)

def get_dict_vectors_of_documents(documents, justTokens=False, scores=None, headers=None):
  init_doc_count = len(documents)/2
  operated_doc_count = 0
  doc_dict_vectors_list = []
  for i in range(len(documents)/2):
    if(operated_doc_count == 400):
      init_doc_count-=400
      operated_doc_count=0
      print str(init_doc_count) + " sets remaining"
    operated_doc_count+=1
    document1, document2 = documents[(2*i)], documents[(2*i)+1]
    sent_1_tokens = process.tokens(document1)
    sent_2_tokens = process.tokens(document2)
    # print sent_1_tokens, sent_2_tokens
    if(justTokens):
      doc_dict_vectors_list.append((sent_1_tokens, sent_2_tokens))
    else:
      dictionary_v = {}
      if headers!= None:
        val = {}
        for idx, header in enumerate(headers):
          val[header] = scores[i][idx]
        # dictionary_v.update(val)
      dictionary_v.update(get_dict_vector_of_2_sentences(sent_1_tokens, sent_2_tokens))
      doc_dict_vectors_list.append(dictionary_v)
  return doc_dict_vectors_list

def appendWordEmbeddings(Doc_Dict_Vectors,TR_S1,TR_S2):
  assert len(Doc_Dict_Vectors)==len(TR_S1)
  for i in range(len(Doc_Dict_Vectors)):
    S1 = TR_S1[i]
    v = {}
    for idx,j in enumerate(S1):
      v['S1_'+str(idx)] = S1[idx]
    # print v
    Doc_Dict_Vectors[i].update(v)
    S2 = TR_S2[i]
    v = {}
    for idx,j in enumerate(S2):
      v['S2_'+str(idx)] = S2[idx]
    # print v
    Doc_Dict_Vectors[i].update(v)
  return Doc_Dict_Vectors

