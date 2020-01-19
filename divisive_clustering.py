import matplotlib.pyplot as plt
import numpy as np


def dfs(src, cluster, visited, n, inp, output):
  cluster.append(inp[src])
  visited.append(src)
  for ix in range(n):
    if output[src][ix] > 0 and ix not in visited:
      dfs(ix, cluster, visited, n, inp, output)

class Graph(): 
  
    def __init__(self, vertices): 
        self.V = vertices 
        self.graph = [[0 for column in range(vertices)]  
                    for row in range(vertices)] 
  
    def printMST(self, parent): 
        output = np.zeros((self.V, self.V))
        for i in range(1, self.V): 
            output[parent[i]][i] = self.graph[i][ parent[i]]
            output[i][parent[i]] = self.graph[i][ parent[i]]
        return output
  
    def minKey(self, key, mstSet): 
        min = float('inf') 
        for v in range(self.V): 
            if key[v] < min and mstSet[v] == False: 
                min = key[v] 
                min_index = v 
        return min_index 
  
    def primMST(self): 
        key = [float('inf')] * self.V 
        parent = [None] * self.V 
        key[0] = 0 
        mstSet = [False] * self.V 
        parent[0] = -1
        for cout in range(self.V): 
            u = self.minKey(key, mstSet) 
            mstSet[u] = True
            for v in range(self.V):  
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]: 
                        key[v] = self.graph[u][v] 
                        parent[v] = u 
  
        return self.printMST(parent) 

def divisive_cluster(n, k):
  if(k>n):
    print("Enter valid value for K!")
    return 
  # ---------------------------------

  plt.plot([10000], [10000])
  inp = plt.ginput(n)

  X = []
  y = []

  for ix, iy in inp:
    X.append(ix)
    y.append(iy)

  plt.scatter(X,y)
  # -------------------------------------
  matrix = np.zeros((n,n))
  for p in range(n):
    for q in range(n):
      if inp[p] is inp[q]:
        continue
      x1,y1 = inp[p]
      x2,y2 = inp[q]
      matrix[p][q] = np.sqrt((x1-x2)**2 + (y1-y2)**2)
  # -------------------------------------
  g = Graph(n)
  g.graph = matrix
  output = g.primMST()
  # -------------------------------------
  #Remove k-1 edges
  for ik in range(0,k-1):
    maxi = -1
    maxX = maxY = -1
    for ix in range(n):
      for iy in range(n):
        if iy>=ix:
          if output[ix,iy] > maxi:
            maxi = output[ix,iy]
            maxX = ix
            maxY = iy
    output[maxX,maxY] = 0
    output[maxY,maxX] = 0
  # -------------------------------------
  # clustering points
  clusters = []
  visited = []

  for i in range(n):
    if i not in visited:
      cluster = []
      dfs(i, cluster, visited, n, inp, output)
      clusters.append(cluster)

  # ---------------------------------------
  # Plot Final Graph
  plt.figure(0)
  for i in range(k):
    X = []
    y = []

    for ix, iy in clusters[i]:
      X.append(ix)
      y.append(iy)
    plt.scatter(X,y)
  plt.show()



divisive_cluster(20, 4)