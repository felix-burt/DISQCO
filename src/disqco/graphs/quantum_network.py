import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class QuantumNetwork():
    def __init__(self, qpu_sizes, qpu_connectivity):
        self.qpu_sizes = qpu_sizes
        self.qpu_connectivity = qpu_connectivity
        self.qpu_graph = self.create_qpu_graph()

    def create_qpu_graph(self):
        qpu_graph = nx.Graph()
        for i, qpu_size in enumerate(self.qpu_sizes):
            qpu_graph.add_node(i, size=qpu_size)
        for i, j in self.qpu_connectivity:
            qpu_graph.add_edge(i, j)
        return qpu_graph
    
    def draw(self,):
        node_sizes = [20*self.qpu_graph.nodes[i]['size'] for i in self.qpu_graph.nodes]
        nx.draw(self.qpu_graph, with_labels=True, node_size=node_sizes)
        plt.show()

    def multi_source_bfs(self, roots, receivers):
        graph = self.qpu_graph

        visited = set()
        parent = dict()   # parent[v] = the node from which we discovered v
        queue = deque()

        # Initialize the queue with all roots
        for r in roots:
            visited.add(r)
            parent[r] = None  # or some sentinel
            queue.append(r)

        # Standard BFS
        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

        # Now reconstruct the edges used to connect each receiver
        chosen_edges = set()
        
        for t in receivers:
            if t not in visited:
                # This receiver is unreachable from any root => no solution for t
                continue
            
            # Walk back from t to some root, collecting edges
            cur = t
            while parent[cur] is not None:  # i.e., cur is not itself a root
                p = parent[cur]
                # Edge is (p,cur) - keep edges in a canonical order, e.g. (min,max)
                chosen_edges.add(tuple(sorted((p, cur))))
                cur = p
        
        return chosen_edges


    def steiner_forest(self, root_config, rec_config):
        root_nodes = [i for i, element in enumerate(root_config) if element == 1]
        rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]
        edges = self.multi_source_bfs(root_nodes, rec_nodes)
        cost = len(edges)
        return edges, cost