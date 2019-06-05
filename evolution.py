import random
import numpy as np
import scipy.spatial.distance as ssd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

code = ['A', 'T', 'C', 'G']

def to_code(a):
    """
    Convert list of integers to corresponding letters
    """
    return [code[i] for i in a]

def dist(a, b):
    """
    Compute distance between two DNA strands
    """
    return sum(i != j for i, j in zip(a, b))

def generate_offspring(parent):
    return [maybe_modify(char) for char in parent]

def maybe_modify(char):
    if np.random.random() < 0.001:
        return np.random.choice(code)
    return char

# Create 3 parents that can contain any of the 4 numbers representing our "code"
p1 = to_code(np.random.randint(4, size=1000))
p2 = to_code(np.random.randint(4, size=1000))
p3 = to_code(np.random.randint(4, size=1000))

# Create offspring
num_generations = 99
max_offspring_per_generation = 1000
current_generation = [p1, p2, p3]

for i in range(num_generations):
    next_generation = []
    for parent in current_generation:
        # Each parent will have between 1 and 3 children
        num_offspring = np.random.randint(3) + 1

        # Generate offspring
        for _ in range(num_offspring):
            child = generate_offspring(parent)
            next_generation.append(child)

    current_generation = next_generation

    # Limit the generation
    random.shuffle(current_generation)
    current_generation = current_generation[:max_offspring_per_generation]

    print("Finished creating generation %d / %d, size = %d" % (i + 2, num_generations + 1, len(current_generation)))


N = len(current_generation)
dist_matrix = np.zeros((N, N))
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        elif j > i:
            a = current_generation[i]
            b = current_generation[j]
            dist_matrix[i,j] = dist(a, b)
        else:
            dist_matrix[i,j] = dist_matrix[j,i]

dist_array = ssd.squareform(dist_matrix)

Z = linkage(dist_array, 'ward')
plt.title("Ward")
dendrogram(Z)
plt.show()

Z = linkage(dist_array, 'single')
plt.title("Single")
dendrogram(Z)
plt.show()

Z = linkage(dist_array, 'complete')
plt.title("Complete")
dendrogram(Z)
plt.show()