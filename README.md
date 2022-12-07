# AI  <br> <br>
BFS <br>
graph = { <br>
  'A' : ['B','C'], <br>
  'B' : ['D', 'E'], <br>
  'C' : ['F'], <br>
  'D' : [], <br>
  'E' : ['F'], <br>
  'F' : [] <br>
} <br>

visited = [] # List to keep track of visited nodes. <br>
queue = []     #Initialize a queue <br>

def bfs(visited, graph, node): <br>
  visited.append(node) <br>
  queue.append(node) <br>
 <br>
  while queue: <br>
    s = queue.pop(0)  <br>
    print (s, end = " ")  <br>

    for neighbour in graph[s]: <br>
      if neighbour not in visited: <br>
        visited.append(neighbour) <br>
        queue.append(neighbour) <br>
 <br>
# Driver Code <br>
bfs(visited, graph, 'A') <br>


OUTPUT <br>

![image](https://user-images.githubusercontent.com/97940767/206164794-bf51c66d-c080-4036-b84f-eb05b929570d.png) <br>





2) TowerOfHanoi <br>


def TowerOfHanoi(n , source, destination, auxiliary): <br>
    if n==1: <br>
        print ("Move disk 1 from source",source,"to destination",destination) <br>
        return <br>
    TowerOfHanoi(n-1, source, auxiliary, destination) <br>
    print ("Move disk",n,"from source",source,"to destination",destination) <br>
    TowerOfHanoi(n-1, auxiliary, destination, source) <br>
         
# Driver code <br>
n = 4 <br>
TowerOfHanoi(n,'A','B','C') <br>
# A, C, B are the name of rods <br>

OUTPUT <br>

![image](https://user-images.githubusercontent.com/97940767/206164983-912efb1f-7dec-408d-9d0f-66efdd4cead7.png) <br>


3)# Using a Python dictionary to act as an adjacency list <br>
graph = { <br>
  '5' : ['3','7'], <br>
  '3' : ['2', '4'], <br>
  '7' : ['8'], <br>
  '2' : [], <br>
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

# Driver Code <br>
print("Following is the Depth-First Search") <br>
dfs(visited, graph, '5') <br>

OUTPUT <br>

![image](https://user-images.githubusercontent.com/97940767/206165384-75cd3897-907f-46a2-bb8a-e656c4523845.png) <br>


