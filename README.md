# AI  <br> <br>
# 1)  PROGRAM TO FIND BFS <br>

graph = { <br>
 '1' : ['2','10'],<br>
 '2' : ['3','8'],<br>
 '3' : ['4'],<br>
 '4' : ['5','6','7'],<br>
 '5' : [],<br>
 '6' : [],<br>
 '7' : [],<br>
 '8' : ['9'],<br>
 '9' : [],<br>
 '10' : []<br>
 }<br>
visited = [] # List for visited nodes.<br>
queue = []     #Initialize a queue<br>

def bfs(visited, graph, node): #function for BFS<br>
  visited.append(node)<br>
  queue.append(node)<br>

  while queue:          # Creating loop to visit each node<br>
    m = queue.pop(0) <br>
    print (m, end = " ") <br>

    for neighbour in graph[m]:<br>
      if neighbour not in visited:<br>
        visited.append(neighbour)<br>
        queue.append(neighbour)<br>

#Driver Code<br>
print("Following is the Breadth-First Search")<br>
bfs(visited, graph, '1')    # function calling<br>


# OUTPUT

![image](https://user-images.githubusercontent.com/97940767/207011954-3b5c91c2-b394-464d-ba0f-01fd68fe6149.png)


# 2)  PROGRAM TO FIND DFS <br>

#Using a Python dictionary to act as an adjacency list <br>
graph = { <br>
 '5' : ['3','7'], <br>
 '3' : ['2', '4'], <br>
 '7' : ['6'], <br>
 '6': [], <br>
 '2' : ['1'], <br>
 '1':[], <br>
 '4' : ['8'], <br>
 '8' : [] <br>
} <br>
visited = set() # Set to keep track of visited nodes of graph. <br>

def dfs(visited, graph, node):  #function for dfs  <br>
    if node not in visited: <br>
        print (node) <br>
        visited.add(node) <br>
        for neighbour in graph[node]: <br>
            dfs(visited, graph, neighbour) <br>

#Driver Code <br>
print("Following is the Depth-First Search") <br>
dfs(visited, graph, '5') <br>

# OUTPUT
![image](https://user-images.githubusercontent.com/97940767/207014283-2d648e14-37a8-414f-a91f-a626984050c7.png)


# 3)  PROGRAM TO FIND BEST FIRST SEARCH <br>


from queue import PriorityQueue<br>
import matplotlib.pyplot as plt<br>
import networkx as nx<br>
#for implementing BFS | returns path having lowest cost<br>
def best_first_search(source, target, n):<br>
 visited = [0] * n<br>
 visited[source] = True<br>
 pq = PriorityQueue()<br>
 pq.put((0, source))<br>
 while pq.empty() == False:<br>
   u = pq.get()[1]<br>
   print(u, end=" ") # the path having lowest cost<br>
   if u == target:<br>
     break<br>
   for v, c in graph[u]:<br>
     if visited[v] == False:<br>
       visited[v] = True<br>
       pq.put((c, v))<br>
       print()<br>
#for adding edges to graph<br>
def addedge(x, y, cost):<br>
 graph[x].append((y, cost))<br>
 graph[y].append((x, cost))<br>

v = int(input("Enter the number of nodes: "))<br>
graph = [[] for i in range(v)] # undirected Graph<br>
e = int(input("Enter the number of edges: "))<br>
print("Enter the edges along with their weights:")<br>
for i in range(e):<br>
 x, y, z = list(map(int, input().split()))<br>
 addedge(x, y, z)<br>
source = int(input("Enter the Source Node: "))<br>
target = int(input("Enter the Target/Destination Node: "))0 3 2<br>
print("Path: ", end = "")<br>
best_first_search(source, target, v)<br>

# OUTPUT

![image](https://user-images.githubusercontent.com/97940767/207552214-8c45668c-402c-42ef-9030-07c3a4e42107.png)<br>

# 4)  PROGRAM TO FIND WATER JUG PROBLEM <br>

from collections import defaultdict <br>
jug1, jug2, aim = 4, 3, 2 <br>
visited = defaultdict(lambda: False) <br>
def waterJugSolver(amt1, amt2): <br>
 if (amt1 == aim and amt2 == 0) or (amt2 == aim and amt1 == 0): <br>
   print(amt1, amt2) <br>
   return True <br>
 if visited[(amt1, amt2)] == False: <br>
   print(amt1, amt2) <br>
   visited[(amt1, amt2)] = True <br>
   return (waterJugSolver(0, amt2) or <br>
 waterJugSolver(amt1, 0) or <br>
 waterJugSolver(jug1, amt2) or <br>
 waterJugSolver(amt1, jug2) or <br>
 waterJugSolver(amt1 + min(amt2, (jug1-amt1)), <br>
 amt2 - min(amt2, (jug1-amt1))) or <br>
 waterJugSolver(amt1 - min(amt1, (jug2-amt2)), <br>
 amt2 + min(amt1, (jug2-amt2)))) <br>
 else: <br>
   return False <br>
print("Steps: ") <br>
waterJugSolver(0, 0) <br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/207561588-ad673d28-78cd-45c6-b1d4-cbaee2c317b0.png)<br>

# 5)  PROGRAM TO FIND TOWER OF HANOI <br>

def TowerOfHanoi(n , source, destination, auxiliary):<br>
    if n==1:
        print ("Move disk 1 from source",source,"to destination",destination)<br>
        return
    TowerOfHanoi(n-1, source, auxiliary, destination)<br>
    print ("Move disk",n,"from source",source,"to destination",destination)<br>
    TowerOfHanoi(n-1, auxiliary, destination, source)

n = 3<br>
TowerOfHanoi(n,'A','B','C')<br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/207562314-34c88549-6fd0-4ff4-b63a-cc88a278869a.png)<br>


# 6)Tic-Tac-Toe Program using

 
#importing all necessary libraries<br>
import numpy as np<br>
import random<br>
from time import sleep<br>
 
#Creates an empty board<br>
 
 
def create_board():<br>
    return(np.array([[0, 0, 0],<br>
                     [0, 0, 0],<br>
                     [0, 0, 0]]))<br>
 
#Check for empty places on board<br>
 
 
def possibilities(board):<br>
    l = []<br>
 
    for i in range(len(board)):<br>
        for j in range(len(board)):<br>
 
            if board[i][j] == 0:<br>
                l.append((i, j))<br>
    return(l)<br>
 <br>
#Select a random place for the player<br><br>
 
 
def random_place(board, player):<br>
    selection = possibilities(board)<br>
    current_loc = random.choice(selection)<br>
    board[current_loc] = player
    return(board)<br>
 
#Checks whether the player has three<br>
#of their marks in a horizontal row<br>
 
 
def row_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True
 <br>
        for y in range(len(board)):<br>
            if board[x, y] != player:<br>
                win = False<br>
                continue<br>
 
        if win == True:<br>
            return(win)<br>
    return(win)<br>
 
#Checks whether the player has three<br>
#of their marks in a vertical row<br>
 
 
def col_win(board, player):<br>
    for x in range(len(board)):<br>
        win = True<br>
 
        for y in range(len(board)):<br>
            if board[y][x] != player:<br>
                win = False<br>
                continue<br>
 
        if win == True:<br>
            return(win)<br>
    return(win)<br>
 
#Checks whether the player has three<br>
#of their marks in a diagonal row<br>
 
 
def diag_win(board, player):<br>
    win = True<br>
    y = 0<br>
    for x in range(len(board)):<br>
        if board[x, x] != player:<br>
            win = False<br>
    if win:<br>
        return win<br>
    win = True<br>
    if win:<br>
        for x in range(len(board)):<br>
            y = len(board) - 1 - x<br>
            if board[x, y] != player:<br>
                win = False<br>
    return win<br>
 
#Evaluates whether there is<br>
#a winner or a tie<br>
 
 
def evaluate(board):<br>
    winner = 0<br>
 <br>
    for player in [1, 2]:<br>
        if (row_win(board, player) or<br>
                col_win(board, player) or<br>
                diag_win(board, player)):<br>
 
            winner = player<br>
 
    if np.all(board != 0) and winner == 0:<br>
        winner = -1<br>
    return winner<br>
 
#Main function to start the game<br>
 
 
def play_game():<br>
    board, winner, counter = create_board(), 0, 1<br>
    print(board)
    sleep(2)<br>
 
    while winner == 0:<br>
        for player in [1, 2]:<br>
            board = random_place(board, player)<br>
            print("Board after " + str(counter) + " move")<br>
            print(board)<br>
            sleep(2)<br>
            counter += 1<br>
            winner = evaluate(board)<br>
            if winner != 0:<br>
                break<br>
    return(winner)<br>
 
 
#Driver Code<br>
print("Winner is: " + str(play_game()))<br>


# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/208868867-8748ba11-90eb-42a1-9405-cf86162efb46.png)


# 7) travellingSalesmanProblem<br>


from sys import maxsize<br>
from itertools import permutations<br>
V = 4<br>

def travellingSalesmanProblem(graph, s):<br>
#store all vertex apart from source vertex<br>
 vertex = []<br>
 for i in range(V):<br>
   if i != s:<br>
    vertex.append(i)<br>

#store minimum weight Hamiltonian Cycle<br>
    min_path = maxsize<br>
    next_permutation=permutations(vertex)<br>
 for i in next_permutation:<br>

#store current Path weight(cost)<br>
        current_pathweight = 0<br>

#compute current path weight<br>
        k = s<br>
        for j in i:<br>
           current_pathweight += graph[k][j]<br>
           k = j<br>
        current_pathweight += graph[k][s]<br>



#Update minimum<br>
        min_path = min(min_path, current_pathweight)<br>
 return min_path<br>

#Driver Code<br>
if __name__ == "__main__":<br>

#matrix representation of graph<br>
 graph = [[0, 10, 15, 20], [10, 0, 35, 25],<br>
          [15, 35, 0, 30], [20, 25, 30, 0]]<br>
s = 0<br>
print(travellingSalesmanProblem(graph, s))<br>

# OUTPUT<br>
![image](https://user-images.githubusercontent.com/97940767/209785691-46ef1be8-b536-4b35-a754-fa809d3a271e.png)


# 8). Write a program to implement the FIND-S Algorithm for finding the most specific 
hypothesis based on a given set of training data samples. Read the training data from a 
.CSV file.


import pandas as pd<br>
import numpy as np<br>
 
#to read the data in the csv file<br>
data = pd.read_csv("ws.csv")<br>
print(data)<br>
 
#making an array of all the attributes<br>
d = np.array(data)[:,:-1]<br>
print("The attributes are: ",d)<br>
 
#segragating the target that has positive and negative examples<br>
target = np.array(data)[:,-1]<br>
print("The target is: ",target)<br>
 
#training function to implement find-s algorithm<br>
def train(c,t):<br>
    for i, val in enumerate(t):<br>
        if val == "Yes":<br>
            specific_hypothesis = c[i].copy()<br>
            break<br>
             
    for i, val in enumerate(c):<br>
        if t[i] == "Yes":<br>
            for x in range(len(specific_hypothesis)):<br>
                if val[x] != specific_hypothesis[x]:<br>
                    specific_hypothesis[x] = '?'<br>
                else:<br>
                    pass<br>
                 
    return specific_hypothesis<br>
 
#obtaining the final hypothesis<br>
print("The final hypothesis is:",train(d,target))<br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/209803977-f7e1194d-8825-49ac-893b-9b6159cdf6d1.png)


# 9). Write a program to implement the Candidate Elimination Algorithm <br>

import csv<br>
with open("ws.csv") as f:<br>
    csv_file=csv.reader(f)<br>
    data=list(csv_file)<br>
    print(data)<br>
    s=data[1][:-1]<br>
    g=[['?' for i in range(len(s))] for j in range(len(s))]<br>
    for i in data:<br>
        if i[-1]=="Yes":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                    s[j]='?'<br>
                    g[j][j]='?'<br>
        elif i[-1]=="No":<br>
            for j in range(len(s)):<br>
                if i[j]!=s[j]:<br>
                    g[j][j]=s[j]<br>
                else:<br>
                    g[j][j]="?"<br>
        print("\nSteps of Candidate Elimination Algorithm",data.index(i)+1)<br>
        print(s)<br>
        print(g)<br>
        gh=[]<br>

for i in g:<br>
 for j in i:<br>
  if j!='?':<br>
   gh.append(i)<br>
   break<br>
print("\nFinal specific hypothesis:\n",s)<br>
print("\nFinal general hypothesis:\n",gh)<br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/210214107-5cafff8e-c76b-48ef-8521-86e249250d48.png)<br>


# 10). Write a program to implement the N QUEEN problem

global N<br>
N = 4<br>
def printSolution(board):<br>
 for i in range(N):<br>
   for j in range(N):<br>
     print (board[i][j], end = " ")<br>
   print()<br>
def isSafe(board, row, col):<br>
 for i in range(col):<br>
   if board[row][i] == 1:<br>
    return False<br>
 for i, j in zip(range(row, -1, -1),range(col, -1, -1)):<br>
   if board[i][j] == 1:<br>
    return False<br>
 for i, j in zip(range(row, N, 1),range(col, -1, -1)):<br>
   if board[i][j] == 1:<br>
    return False<br>
 return True<br>
def solveNQUtil(board, col):<br>
 if col >= N:<br>
   return True<br>
 for i in range(N):<br>
   if isSafe(board, i, col):<br>
     board[i][col] = 1 <br>
     if solveNQUtil(board, col + 1) == True:<br>
       return True<br>
     board[i][col] = 0<br>
 return False<br>
def solveNQ():<br>
 board = [ [0, 0, 0, 0],<br>
 [0, 0, 0, 0],<br>
 [0, 0, 0, 0],<br>
 [0, 0, 0, 0] ]<br>
 if solveNQUtil(board, 0) == False:<br>
   print ("Solution does not exist")<br>
   return False<br>
 printSolution(board)<br>
 return True<br>
solveNQ()<br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/210214323-1a8ec27b-1c6c-453a-89df-f58f77882c1f.png)<br>

# 11) Write a program to implement A star algorithm<br>


def aStarAlgo(start_node, stop_node):<br>
         
        open_set = set(start_node) <br>
        closed_set = set()<br>
        g = {} #store distance from starting node<br>
        parents = {}# parents contains an adjacency map of all nodes<br>
 
        #ditance of starting node from itself is zero<br>
        g[start_node] = 0<br>
        #start_node is root node i.e it has no parent nodes<br>
        #so start_node is set to its own parent node<br>
        parents[start_node] = start_node<br>
         
         
        while len(open_set) > 0:<br>
            n = None<br>
 
            #node with lowest f() is found<br>
            for v in open_set:<br>
                if n == None or g[v] + heuristic(v) < g[n] + heuristic(n):<br>
                    n = v<br>
             
                     
            if n == stop_node or Graph_nodes[n] == None:<br>
                pass<br>
            else:<br>
                for (m, weight) in get_neighbors(n):<br>
                    #nodes 'm' not in first and last set are added to first<br>
                    #n is set its parent<br>
                    if m not in open_set and m not in closed_set:<br>
                        open_set.add(m)<br>
                        parents[m] = n<br>
                        g[m] = g[n] + weight<br>
                         
     
                    #for each node m,compare its distance from start i.e g(m) to the<br>
                    #from start through n node<br>
                    else:<br>
                        if g[m] > g[n] + weight:<br>
                            #update g(m)<br>
                            g[m] = g[n] + weight<br>
                            #change parent of m to n<br>
                            parents[m] = n<br>
                             
                            #if m in closed set,remove and add to open<br>
                            if m in closed_set:<br>
                                closed_set.remove(m)<br>
                                open_set.add(m)<br>
 
            if n == None:
                print('Path does not exist!')<br>
                return None<br>
 
            # if the current node is the stop_node<br>
            # then we begin reconstructin the path from it to the start_node<br>
            if n == stop_node:<br>
                path = []<br>
 
                while parents[n] != n:<br>
                    path.append(n)<br>
                    n = parents[n]<br>
 
                path.append(start_node)<br>
 
                path.reverse()<br>
<br>
                print('Path found: {}'.format(path))<br>
                return path<br>
 
 
            # remove n from the open_list, and add it to closed_list<br>
            # because all of his neighbors were inspected<br>
            open_set.remove(n)<br>
            closed_set.add(n)<br>
 
        print('Path does not exist!')<br>
        return None<br>
         
#define fuction to return neighbor and its distance<br>
#from the passed node<br>
def get_neighbors(v):<br>
    if v in Graph_nodes:<br>
        return Graph_nodes[v]<br>
    else:<br>
        return None<br>
#for simplicity we ll consider heuristic distances given<br>
#and this function returns heuristic distance for all nodes<br>
def heuristic(n):<br>
        H_dist = {<br>
            'A': 11,<br>
            'B': 6,<br>
            'C': 99,<br>
            'D': 1,<br>
            'E': 7,<br>
            'G': 0,<br>
             
        }<br>
 
        return H_dist[n]<br>
 
#Describe your graph here  <br>
Graph_nodes = {<br>
    'A': [('B', 2), ('E', 3)],<br>
    'B': [('C', 1),('G', 9)],<br>
    'C': None,<br>
    'E': [('D', 6)],<br>
    'D': [('G', 1)],<br>
     
}<br>
aStarAlgo('A', 'G')<br>

# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/210525071-b2322718-8d4f-44f5-8ec4-7dc980fb7fbc.png)



# BARATHI MAAM PROGRAM

# 1)Write a program to implement Decision Tree classifier to find accuracy for training and test fruit data set.

import pandas as pd

fruits = pd.read_table('fruit_data_with_colors.txt')
feature_names = ['mass', 'width', 'height', 'color_score']   #all attributes

X = fruits[feature_names]
y = fruits['fruit_label']                                # y only label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Decision Tree classifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(X_train, y_train)

print('Accuracy of Decision Tree classifier on training set: {:.2f}'.format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clf.score(X_test, y_test)))

# OUTPUT<br>


![image](https://user-images.githubusercontent.com/97940767/212030448-c9cbacdf-5bae-4994-90f2-c7ad483f51e2.png)

# 2) Write a program to implement K-means clustering using random samples
from copy import deepcopy

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt

#Set three centers, the model should predict similar results
center_1 = np.array([1,1])
center_2 = np.array([5,5])
center_3 = np.array([8,1])

#Generate random data and center it to the three centers
data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200,2) + center_2
data_3 = np.random.randn(200,2) + center_3
data = np.concatenate((data_1, data_2, data_3), axis = 0)
plt.scatter(data[:,0], data[:,1], s=7)

#Number of clusters
k = 3

#Number of training data
n = data.shape[0]

#Number of features in the data
c = data.shape[1]

#Generate random centers, here we use sigma and mean to ensure it represent the whole data
mean = np.mean(data, axis = 0)
std = np.std(data, axis = 0)
centers = np.random.randn(k,c)*std + mean

#Plot the data and the centers generated as random
plt.scatter(data[:,0], data[:,1], s=9,color='c')
plt.scatter(centers[:,0], centers[:,1], marker='*', c='g',s=150)
plt.show()


# OUTPUT<br>

![image](https://user-images.githubusercontent.com/97940767/212031816-a05f8f8d-326a-440c-8424-b8861c392ad5.png)





