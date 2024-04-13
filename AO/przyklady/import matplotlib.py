import matplotlib.pyplot as plt

# Constants for steel
YIELD_STRENGTH = 250000000  # in Pascals
MOMENT_OF_INERTIA = 0.00001  # in m^4, for an H200 beam

# Beam lengths in meters
lengths = range(1, 11)

# Calculate the maximum load for each length
max_loads = [YIELD_STRENGTH * MOMENT_OF_INERTIA / (0.5 * length) for length in lengths]

# Create a plot
plt.plot(lengths, max_loads)
plt.xlabel('Beam Length (m)')
plt.ylabel('Maximum Load (N)')
plt.title('Maximum Load vs Beam Length for an H200 Beam')
plt.grid(True)
plt.show()