import glob
import itertools


txts = []
for file in glob.glob("txt/*.txt"):
    with open(file, "r") as doc:
        corpus.append(doc.read())

txts.insert(0, "web development design")

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer()
term_vectors = vectorizer.fit_transform(txts)


tdm = np.array(term_vectors.toarray())
# 2.3) if it has to be term as i doc as j
print tdm.transpose()


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(use_idf=False, smooth_idf=False)
# for 3.1a)
tf = transformer.fit_transform(term_vectors.toarray())



transformer = TfidfTransformer(use_idf=True, smooth_idf=False)
# for 3.1b)
tfidf = transformer.fit_transform(term_vectors.toarray())

from sklearn.metrics.pairwise import linear_kernel

# i added the query "web development design"
cosine_similarities_tf = linear_kernel(tf[0:1], tf).flatten()
print cosine_similarities_tf.argsort()[:-7:-1]

cosine_similarities_tfidf = linear_kernel(tfidf[0:1], tfidf).flatten()
print cosine_similarities_tfidf.argsort()[:-7:-1]