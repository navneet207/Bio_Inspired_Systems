import numpy as np
import time

# Define the update rule for the cellular automaton (e.g., Rule 30 for simplicity)
def rule30(left, center, right):
    """Rule 30: Update rule for each cell in the automaton."""
    return left ^ (center or right)

# Function to update the grid in a single process (no parallelism)
def update_grid(grid):
    """Update the entire grid using the Rule 30."""
    new_grid = np.zeros_like(grid)  # Create a new grid to store updated values

    # Iterate over the grid and apply the rule for each cell
    for i in range(len(grid)):
        left = grid[i - 1] if i > 0 else 0  # Left neighbor (0 if out of bounds)
        center = grid[i]  # Current cell
        right = grid[i + 1] if i < len(grid) - 1 else 0  # Right neighbor (0 if out of bounds)
        new_grid[i] = rule30(left, center, right)
    
    return new_grid

# Function to print the grid in numeric format
def print_grid(grid):
    """Print the grid in numeric format."""
    print("".join([str(cell) for cell in grid]))

if __name__ == "__main__":
    # Initial state (1D array with binary values, 0 or 1)
    initial_state = [0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0]
    
    # Number of steps to run the automaton
    steps = 10
    
    print("Initial State:")
    print_grid(initial_state)
    print("\nRunning Cellular Automaton...\n")
    
    # Simulate the cellular automaton for the given number of steps
    current_state = np.array(initial_state)  # Convert to numpy array for easy manipulation
    for step in range(steps):
        current_state = update_grid(current_state)  # Update the grid in each step
        print(f"Step {step + 1}:")
        print_grid(current_state)
        time.sleep(0.5)  # Add delay for better visualization in console
