# Apply appropriate Optimization Algorithm for Analyzing protein-protein interaction

import numpy as np
import networkx as nx
import random

# ------------------------------
# Step 1. Load or simulate a PPI network
# ------------------------------
def load_ppi_network():
    """
    Example: creates a synthetic PPI network.
    In real use, load from BioGRID, STRING, or IntAct data.
    """
    G = nx.barabasi_albert_graph(100, 3)  # synthetic scale-free PPI-like graph
    return G


# ------------------------------
# Step 2. Fitness function — Modularity of clustering
# ------------------------------
def fitness(G, clusters):
    """
    Compute modularity of the clustering.
    clusters: array of cluster IDs for each node (same length as G.nodes)
    """
    community_dict = {}
    for node, cluster_id in enumerate(clusters):
        community_dict.setdefault(cluster_id, []).append(node)

    communities = list(community_dict.values())
    return nx.algorithms.community.quality.modularity(G, communities)


# ------------------------------
# Step 3. Cuckoo Search Algorithm
# ------------------------------
def cuckoo_search_ppi(G, n_nests=20, pa=0.25, alpha=1.0, max_iter=30):
    n = len(G.nodes)
    nests = [np.random.randint(0, 5, n) for _ in range(n_nests)]  # 5 possible clusters
    fitness_values = [fitness(G, nest) for nest in nests]

    best_index = np.argmax(fitness_values)
    best_nest = nests[best_index].copy()
    best_fitness = fitness_values[best_index]

    for iteration in range(max_iter):
        for i in range(n_nests):
            # Lévy flight (random walk)
            step_size = alpha * np.random.randn(n)
            new_nest = nests[i] + step_size.astype(int)
            new_nest = np.abs(new_nest) % 5  # keep cluster IDs within range
            new_fitness = fitness(G, new_nest)

            # Replace if better
            if new_fitness > fitness_values[i]:
                nests[i] = new_nest
                fitness_values[i] = new_fitness

        # Abandon some nests (discovery probability)
        for i in range(n_nests):
            if random.random() < pa:
                nests[i] = np.random.randint(0, 5, n)
                fitness_values[i] = fitness(G, nests[i])

        # Update the global best
        best_index = np.argmax(fitness_values)
        if fitness_values[best_index] > best_fitness:
            best_fitness = fitness_values[best_index]
            best_nest = nests[best_index].copy()

        print(f"Iteration {iteration+1}/{max_iter}, Best modularity = {best_fitness:.4f}")

    return best_nest, best_fitness


# ------------------------------
# Step 4. Run optimization
# ------------------------------
if __name__ == "__main__":
    G = load_ppi_network()
    best_clusters, best_score = cuckoo_search_ppi(G)
    print("\nFinal best modularity:", best_score)
