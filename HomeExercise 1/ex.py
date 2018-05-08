

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(txts)

# 2.3) term as i doc as j
tdm = np.array(X.toarray()).transpose()
print tdm