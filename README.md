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

