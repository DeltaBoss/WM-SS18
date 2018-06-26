### Task 1 ###
from networkx import nx
import random
from collections import defaultdict

## generate (directed) GNP random graph
G = nx.MultiDiGraph(nx.gnp_random_graph(200, 0.4, directed=True))
visit_freq_map = defaultdict(int) # to count node visits

for i in range(100):
    H = G.copy()
    currNode = 0
    while(random.random() >= 0.2):
        new_link = list(H.edges(currNode))[random.randint(0, len(H.edges(currNode))-1)]
        H.add_edge(new_link[0], new_link[1])  # to increase visit probability
        currNode = new_link[1]
        visit_freq_map[currNode] += 1

# print out nodes by frequency of visit in descending order
print("nodes by frequency of visit:")
print(sorted(visit_freq_map.items(), key=lambda kv: kv[1], reverse=True))

### Task 2 ###

from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import SVD
from surprise import NMF

data = Dataset.load_builtin('ml-100k')

# 5-fold cross validation results for 5 algorithms 
cross_validate(SVD(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(NMF(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(surprise.prediction_algorithms.baseline_only.BaselineOnly(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(surprise.prediction_algorithms.knns.KNNBasic(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
cross_validate(surprise.prediction_algorithms.random_pred.NormalPredictor(), data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Comparison on RMSE, MAE with 5-fold cv result: SVD > Baseline > NMF > k-NN > Random
