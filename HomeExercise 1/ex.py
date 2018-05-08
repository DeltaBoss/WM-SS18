import os
import glob
import itertools


txts = []
titles = []
for file in glob.glob("txt/*.txt"):
  titles.append(os.path.basename(file))
  with open(file, "r") as doc:
    txts.append(doc.read())

txts.insert(0, "web development design")
titles.insert(0, "query")

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

# added the query "web development design" as the first row 
cosine_similarities_tf = linear_kernel(tf[0:1], tf).flatten()
top5_tf = cosine_similarities_tf.argsort()[:-7:-1]
for i in range(0, 6):
  print titles[top5_tf[i]]


cosine_similarities_tfidf = linear_kernel(tfidf[0:1], tfidf).flatten()
top5_tfidf = cosine_similarities_tfidf.argsort()[:-7:-1]
for i in range(0, 6):
  print titles[top5_tfidf[i]]

#3.2
vocab = vectorizer.get_feature_names()
vocdict = dict(itertools.izip_longest(vocab, range(len(vocab))))

i = 0
probs = {}
for row in term_vectors.toarray():
  total = (1.0/max(1, np.sum(row)))
  p = row[vocdict['web']]*total + row[vocdict['development']]*total + row[vocdict['design']]*total
  probs[i] = p
  i += 1

import operator
sorted_probs = sorted(probs.items(), key=operator.itemgetter(1), reverse=True)

for i in range(0, 6):
  print titles[sorted_probs[i][0]]
  #print txts[sorted_probs[i][0]]
