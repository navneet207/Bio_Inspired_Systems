import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target
num_features = X.shape[1]

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Parameters
grid_rows = 5
grid_cols = 5
iterations = 15

# Initialize population: each cell is a binary mask for feature selection
population = np.random.choice([0, 1], size=(grid_rows, grid_cols, num_features))

# Fitness: classification error (lower is better)
def fitness(mask):
    if np.sum(mask) == 0:  # avoid empty feature subset
        return 1.0  # worst fitness
    
    selected_features = np.where(mask == 1)[0]
    
    # Train logistic regression on selected features
    model = LogisticRegression(max_iter=500, solver='liblinear')
    model.fit(X_train[:, selected_features], y_train)
    
    # Predict and calculate error
    y_pred = model.predict(X_val[:, selected_features])
    error = 1 - accuracy_score(y_val, y_pred)
    return error

# Evaluate the whole population
def evaluate_population(pop):
    fitness_grid = np.zeros((grid_rows, grid_cols))
    for i in range(grid_rows):
        for j in range(grid_cols):
            fitness_grid[i, j] = fitness(pop[i, j])
    return fitness_grid

# Get Moore neighborhood with wrap-around
def get_neighbors(i, j, rows, cols):
    neighbors = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            ni = (i + di) % rows
            nj = (j + dj) % cols
            neighbors.append((ni, nj))
    return neighbors

best_solution = None
best_fitness = float('inf')

for it in range(iterations):
    fitness_grid = evaluate_population(population)
    
    # Track global best
    min_idx = np.unravel_index(np.argmin(fitness_grid), fitness_grid.shape)
    if fitness_grid[min_idx] < best_fitness:
        best_fitness = fitness_grid[min_idx]
        best_solution = population[min_idx].copy()

    # Update population
    new_population = np.copy(population)
    for i in range(grid_rows):
        for j in range(grid_cols):
            neighbors = get_neighbors(i, j, grid_rows, grid_cols)
            best_neighbor_mask = population[i, j]
            best_neighbor_fitness = fitness_grid[i, j]
            
            for (ni, nj) in neighbors:
                if fitness_grid[ni, nj] < best_neighbor_fitness:
                    best_neighbor_fitness = fitness_grid[ni, nj]
                    best_neighbor_mask = population[ni, nj]

            # Adopt better neighbor mask if any
            if best_neighbor_fitness < fitness_grid[i, j]:
                new_population[i, j] = best_neighbor_mask
    
    population = new_population

    print(f"Iteration {it+1}/{iterations}, Best Fitness (Error): {best_fitness:.4f}")

print("\nBest feature subset found (binary mask):")
print(best_solution)
print(f"Number of selected features: {np.sum(best_solution)}")
print(f"Classification accuracy: {1 - best_fitness:.4f}")
