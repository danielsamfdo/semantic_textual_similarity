from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import lib.utilities as utility
from sklearn import preprocessing
import models.w2vec as w2vec
from scipy.sparse import hstack
import copy
from sklearn.externals import joblib
MODEL_FILE = 'lr_model.pkl'

def predict(sentence_1, sentence_2, w2vec_model, use_w2_vec_model=True):
  training_documents = [sentence_1, sentence_2]
  v = DictVectorizer(sparse=True)
  v.vocabulary_ = joblib.load("dict_vectorizer_vocab.pkl")
  v.feature_names_ = joblib.load("dict_vectorizer_fnames.pkl")
  lm = joblib.load(MODEL_FILE)
  Doc_Dict_Vectors = utility.get_dict_vectors_of_documents(training_documents,False, None, None)
  # print Doc_Dict_Vectors
  if(use_w2_vec_model):
      TR_S1,TR_S2 = w2vec.w2vec_for_pair_of_docs(training_documents, w2vec_model)
      Doc_Dict_Vectors = utility.appendWordEmbeddings(Doc_Dict_Vectors,TR_S1,TR_S2)
  X = v.transform(Doc_Dict_Vectors)
  # min_max_scaler = preprocessing.StandardScaler(with_mean=False)
  # X_train_minmax = min_max_scaler.transform(X)
  X_normalized = preprocessing.normalize(X, norm='l2')
  predicted_answers = lm.predict(X_normalized)
  print predicted_answers[0], len(predicted_answers)

def linear_regression(training_documents, test_documents, training_answers,  test_answers, load, w2vec_model, use_w2_vec_model):
  lm = LinearRegression()
  # doc_dict_vectors_list = []
  # Corpus = []
  # vectorizer = CountVectorizer(min_df=1)
  # for i in range(len(documents)/2):
  #   Corpus.append(documents[(2*i)] + " " + documents[(2*i)+1])  
  # X = vectorizer.fit_transform(Corpus)
  # vectorizer.transform(['Something completely new.']).toarray()
  # pX = v.fit_transform(D2)

  if(load):
    Doc_Dict_Vectors = utility.load_weights("weights/Feature_Vector.dat")
    Test_doc_dict_vectors = utility.load_weights("weights/Test_Feature_Vector.dat")
    X = utility.load_weights("weights/Train_X.dat")
    pX = utility.load_weights("weights/Test_X.dat")
  else:
    scores = utility.load_weights("weights/jc_cc_scores.dat")
    headers = utility.load_weights("weights/headers_scores_jc_cc.dat")
    tr_scores = scores[:len(training_answers)]
    te_scores = scores[len(training_answers):]
    Doc_Dict_Vectors = utility.get_dict_vectors_of_documents(training_documents,None,tr_scores,headers)
    Test_doc_dict_vectors = utility.get_dict_vectors_of_documents(test_documents,None,te_scores,headers)
    if(use_w2_vec_model):
      TR_S1,TR_S2 = w2vec.w2vec_for_pair_of_docs(training_documents, w2vec_model)
      TE_S1,TE_S2 = w2vec.w2vec_for_pair_of_docs(test_documents, w2vec_model)
      # print len(Doc_Dict_Vectors[0])
      Doc_Dict_Vectors = utility.appendWordEmbeddings(Doc_Dict_Vectors,TR_S1,TR_S2)
      Test_doc_dict_vectors = utility.appendWordEmbeddings(Test_doc_dict_vectors,TE_S1,TE_S2)
      # print len(Doc_Dict_Vectors[0])
    utility.save_weights("Feature_Vector.dat",Doc_Dict_Vectors)
    utility.save_weights("Test_Feature_Vector.dat",Test_doc_dict_vectors)
    v = DictVectorizer(sparse=True)
    X = v.fit_transform(Doc_Dict_Vectors)
    joblib.dump(v.vocabulary_, "dict_vectorizer_vocab.pkl") 
    joblib.dump(v.feature_names_, "dict_vectorizer_fnames.pkl") 
    pX = v.transform(Test_doc_dict_vectors)
    # if(use_w2_vec_model):
      # print X
      # print TR_S1
      # print TR_S2
      # X = hstack([X,TR_S1,TR_S2]).toarray()
      # print X
      # pX = hstack([pX,TE_S1,TE_S2]).toarray()
    utility.save_weights("Train_X.dat",X)
    utility.save_weights("Test_X.dat",pX)
    # print training_answers
  # min_max_scaler = preprocessing.MaxAbsScaler()
  # X_train_minmax = min_max_scaler.fit_transform(X)
  X_normalized = preprocessing.normalize(X, norm='l2')
  lm.fit(X_normalized, training_answers)
  joblib.dump(lm, MODEL_FILE) 
  predicted_answers = lm.predict(X_normalized)
  answers = []
  for i in predicted_answers:
    if(i<0):
      # print "came in"
      answers.append(0)
    elif(i>5):
      # print "came in *** "
      answers.append(5)
    else:
      answers.append(i)
  # print answers
  print "Error in Estimation of Linear Regression - Training : "+str(utility.evaluate(training_answers,answers))
  train_ans = copy.copy(answers)
  pX_normalized = preprocessing.normalize(pX, norm='l2')
  # pX_test_minmax = min_max_scaler.fit_transform(pX)
  predicted_answers = lm.predict(pX_normalized)
  answers = []
  for i in predicted_answers:
    if(i<0):
      # print "came in"
      answers.append(0)
    elif(i>5):
      # print "came in *** "
      answers.append(5)
    else:
      answers.append(i)
  # print answers
  print "Error in Estimation of Linear Regression - Testing : "+str(utility.evaluate(test_answers,answers))

  print "Total Correlation Measure Pearson LR " + str(utility.evaluate_pearsons_coefficient(test_answers, answers))
