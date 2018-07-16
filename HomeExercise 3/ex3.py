import re
import networkx as nx
import numpy as np
import random as r

#loss functions
#args: node u>0, node v>0, (probabilistic) adjacency matrix
def loss1(u, v, adMat):
    tmp = np.matmul(adMat[:,u-1].T, adMat[:,v-1]).item(0,0) - adMat.item(u-1,v-1)
    return tmp**2

def loss2(u, v, adMat):
    tmp1 = np.matmul(adMat[:,u-1].T, adMat[:,v-1]).item(0,0) - adMat.item(u-1,v-1)
    tmp2 = np.matmul(adMat[:,u-1].T, adMat[:,v-1]).item(0,0) - np.matmul(adMat,adMat).item(u-1,v-1)
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

#get probabilistic adjacency matrix
adMat = nx.adjacency_matrix(G).todense().astype(float)
rows = adMat.shape[0]
cols = adMat.shape[1]

i = 0
while i < rows:
    s = adMat[i,:].sum()
    j = 0
    while j < cols:
        adMat[i,j] /= s
        j += 1
    i += 1

#print(loss2(1,2,adMat))

#do gradient descent
learnRate = 0.001
iterations = 0#5000

i = 0
while i < iterations:
    u = r.randint(1,rows)
    v = r.randint(1,cols)

    pass
    
    i += 1
