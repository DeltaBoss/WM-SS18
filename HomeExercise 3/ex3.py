import re
import networkx as nx
import numpy as np
import random as r
import math

#loss functions
#args: node u>0, node v>0, representation matrix, adjacency matrix
def loss1(u, v, emMat, adMat):
    tmp = np.matmul(emMat[:,u-1].T, emMat[:,v-1]).item(0,0) - adMat.item(u-1,v-1)
    return tmp**2

def loss2(u, v, emMat, adMat):
    tmp1 = np.matmul(emMat[:,u-1].T, emMat[:,v-1]).item(0,0) - adMat.item(u-1,v-1)
    tmp2 = np.matmul(emMat[:,u-1].T, emMat[:,v-1]).item(0,0) - np.matmul(adMat,adMat).item(u-1,v-1)
    return tmp1**2 + tmp2**2

#get data from file
fs = open('out.arenas-jazz')
G = nx.Graph()
for line in fs:
    if re.match("%.*",line):
        continue
    else:
        u,v = line.strip().split()
        G.add_edge(u,v)

#get adjacency matrix
adMat = nx.adjacency_matrix(G).todense().astype(float)
rows = adMat.shape[0]
cols = adMat.shape[1]

#use probabilistic adjacency matrix because standard one results in too large numbers to compute
if True:
    i = 0
    while i < rows:
        s = adMat[i,:].sum()
        j = 0
        while j < cols:
            adMat[i,j] /= s
            j += 1
        i += 1

#create initial embedding matrix as copy of adjacency matrix
emMat = adMat.copy()

#do gradient descent
learnRate = 0.001
iterations = 5000

def calcRepresentation(learnRate, iterations, adMat, emMat, lossFunc):
    tmpMat = emMat.copy()
    i = 0
    while i < iterations:
        u = r.randint(1,cols)
        v = r.randint(1,cols)
        if u == v:
            continue
        
        grad = lossFunc(u,v,tmpMat,adMat)
        tmpMat = tmpMat - learnRate*grad
        
        i += 1
    return tmpMat

rep1 = calcRepresentation(learnRate, iterations, adMat, emMat, loss1)
rep2 = calcRepresentation(learnRate, iterations, adMat, emMat, loss2)

#get 5 most similar nodes to node 1 (0 doesnt exist yo)
#returns list of tuples (distance, node nr)
def findSimilarTo(emMat):
    dist = []
    leng = emMat.shape[0]
    it = emMat.shape[1]
    i = 1
    while i < it:
        j = 0
        summ = 0
        while j < leng:
            summ += (emMat.item(0,j) - emMat.item(i,j))**2
            j += 1
        dist.append((math.sqrt(summ),i))
        i += 1
    return sorted(dist)[:5]

print(findSimilarTo(rep1))
print(findSimilarTo(rep2))
