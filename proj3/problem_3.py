import numpy as np

# Original points (x, y)
original_points = np.array([
    [1, 1],     # A
    [1.5, 0.5], # B
    [2, 1],     # C
    [2.5, 2]    # D
])

# Transformed points (x', y')
transformed_points = np.array([
    [-0.9, 0.8],   # A'
    [-0.1, 1.3],   # B'
    [-0.4, 1.9],   # C'
    [-1.25, 2.55]  # D'
])

m_matrix = np.array([
    ["m11","m12"],
    ["m21","m22"]
])

x_matrix = np.array([
    ["x","y"],
])

x1_matrix = np.array([
    ["x'","y'"]
])

#print original equation
print(f"Original equation")
print(f"{m_matrix} x {x_matrix} = {x1_matrix}")
print("")
#rewrite equations and print
print("Expanded set of linear equations")
print("x*m_11 + y*m_12 + 0*m_21 + 0*m_22 = x'")
print("0*m_11 + 0*m_12 + x*m_21 + y*m_22 = y'")
print("")

# Prepare the matrices for least squares
#8 total equations

Q = []
b = []

# Loop through each point pair to construct the equations
for i in range(len(original_points)):
    x, y = original_points[i]
    x_prime, y_prime = transformed_points[i]
    
    # Create equations for Q and b
    Q.append([x, y, 0, 0])       
    b.append(x_prime)             
    
    Q.append([0, 0, x, y])      
    b.append(y_prime)           

# Convert to numpy arrays
Q = np.array(Q)
b = np.array(b)

# Print the resulting matrices
print("Matrix Q:")
print(Q)

print("\nVector b:")
print(b)
print("")

m, residuals, rank, s = np.linalg.lstsq(Q, b, rcond=None)

# Reshape the output m into a 2x2 matrix M
M = m.reshape(2, 2)

# Print the resulting matrix M
print("Matrix M:")
print(M)
