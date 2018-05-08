

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer()
termvectors = vectorizer.fit_transform(txts)


tdm = np.array(X.toarray())
# 2.3) if it has to be term as i doc as j
print tdm.transpose()


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(use_idf=False, smooth_idf=False)
tfidf = transformer.fit_transform(tdm)

# for 3.1a)
#print tfidf.toarray()

transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
tfidf = transformer.fit_transform(tdm)
# for 3.1b)
#print tfidf.toarray()