import heapq

class Node:
   def __init__(self, state, g=0, h=0, parent=None):
       self.state = state
       self.g = g
       self.h = h
       self.f = g + h
       self.parent = parent

   def __lt__(self, other):
       return self.f < other.f

def ida_star(start, goal, heuristic, get_neighbors):
   def search(node, limit):
       if node.f > limit:
           return node.f
       if node.state == goal:
           return node
       
       min_cost = float('inf')
       for neighbor in get_neighbors(node.state):
           g = node.g + 1
           h = heuristic(neighbor, goal)
           child_node = Node(neighbor, g, h, node)
           
           result = search(child_node, limit)
           if isinstance(result, Node):
               return result
           min_cost = min(min_cost, result)
       
       return min_cost

   start_node = Node(start, g=0, h=heuristic(start, goal))
   limit = start_node.f
   
   while True:
       result = search(start_node, limit)
       if isinstance(result, Node):
           path = []
           while result is not None:
               path.append(result.state)
               result = result.parent
           path.reverse()
           return path
       elif result == float('inf'):
           return None
       else:
           limit = result

def heuristic(state, goal):
   dist = 0
   for i in range(len(state)):
       if state[i] != goal[i]:
           goal_index = goal.index(state[i])
           dist += abs(i // 3 - goal_index // 3) + abs(i % 3 - goal_index % 3)
   return dist

def get_neighbors(state):
   neighbors = []
   zero_pos = state.index(0)
   directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
   
   row, col = zero_pos // 3, zero_pos % 3
   for dr, dc in directions:
       new_row, new_col = row + dr, col + dc
       if 0 <= new_row < 3 and 0 <= new_col < 3:
           new_pos = new_row * 3 + new_col
           new_state = list(state)
           new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
           neighbors.append(tuple(new_state))
   
   return neighbors
