import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Create Example Time Series Data
np.random.seed(42)  # For reproducibility
n = 100  # Number of data points
time = np.arange(1, n+1)
data = 0.6 * np.roll(np.random.randn(n), 1) + np.random.randn(n)  # Autoregressive pattern

# Step 2: Plot the Time Series
plt.figure(figsize=(10, 4))
plt.plot(time, data, label="Time Series Data")
plt.title("Synthetic Time Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
#plt.show()

# Step 3: Compute and Plot ACF and PACF
fig, ax = plt.subplots(2, 1, figsize=(10, 8))

# Plot ACF (AutoCorrelation Function)
plot_acf(data, ax=ax[0], lags=1, title="Autocorrelation Function (ACF)")

# Plot PACF (Partial Autocorrelation Function)
plot_pacf(data, ax=ax[1], lags=1, title="Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()
