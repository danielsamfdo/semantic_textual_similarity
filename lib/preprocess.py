from nltk.stem import WordNetLemmatizer
#from functools32 import lru_cache
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
import nltk

def tokens(sentence):
  return word_tokenize(sentence)

def lemmatize(tokens):
  wnl = WordNetLemmatizer()
  return [wnl.lemmatize(token) for token in tokens]

def postags(tokens):
  return nltk.pos_tag(tokens)

def sentences_array(filename): 
  with open(fname) as f:
    lines = f.readlines()
  for line in lines:
    if len(line.split('\t')) ==2 :
        print line

# http://stackoverflow.com/questions/16181419/is-it-possible-to-speed-up-wordnet-lemmatizer
# wnl = WordNetLemmatizer()
# lemmatize = lru_cache(maxsize=50000)(wnl.lemmatize)
