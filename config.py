import nltk
from nltk.corpus import stopwords
from nltk import SnowballStemmer
ppdbDict = {}
ppdbSim = 0.9
theta1 = 0.9
punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'',')', '-rrb-']
stemmer = SnowballStemmer('english')
punctuations = ['(','-lrb-','.',',','-','?','!',';','_',':','{','}','[','/',']','...','"','\'',')', '-rrb-']
stopwords = stopwords.words('english')