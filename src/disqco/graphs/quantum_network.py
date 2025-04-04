import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class QuantumNetwork():
    def __init__(self, qpu_sizes, qpu_connectivity = None):
        self.qpu_sizes = qpu_sizes
        if qpu_connectivity is None:
            self.qpu_connectivity = [(i, j) for i in range(len(qpu_sizes)) for j in range(i+1, len(qpu_sizes))]
        else:
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
        node_colors = [self.qpu_graph.nodes[i]['color'] if 'color' in self.qpu_graph.nodes[i] else 'green' for i in self.qpu_graph.nodes]
        nx.draw(self.qpu_graph, with_labels=True, node_size=node_sizes, node_color=node_colors)
        plt.show()

    def multi_source_bfs(self, roots, receivers):
        graph = self.qpu_graph

        visited = set()
        parent = dict()   
        queue = deque()

        for r in roots:
            visited.add(r)
            parent[r] = None 
            queue.append(r)

        while queue:
            u = queue.popleft()
            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    parent[v] = u
                    queue.append(v)

        chosen_edges = set()
        
        for t in receivers:
            if t not in visited:
                continue

            cur = t
            while parent[cur] is not None: 
                p = parent[cur]
                chosen_edges.add(tuple(sorted((p, cur))))
                cur = p
        
        return chosen_edges

    def steiner_forest(self, root_config, rec_config, node_map=None):
        if node_map is not None:
            root_nodes = [node_map[i] for i in range(len(root_config)) if root_config[i] == 1]
            rec_nodes = [node_map[i] for i in range(len(rec_config)) if rec_config[i] == 1]
        else:
            root_nodes = [i for i, element in enumerate(root_config) if element == 1]
            rec_nodes = [i for i, element in enumerate(rec_config) if element == 1]
        edges = self.multi_source_bfs(root_nodes, rec_nodes)
        cost = len(edges)
        
        return edges, cost