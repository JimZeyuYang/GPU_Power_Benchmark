import numpy as np

# Define the function values and the corresponding x values
x = np.array([0, 1, 2, 2, 4, 4, 6])
y = np.array([0, 0, 0, 1, 1, 0, 0])

integral_value = np.trapz(y, x)

print(integral_value)
