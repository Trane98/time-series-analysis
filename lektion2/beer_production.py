#############################################################################
#
# TSA L2: Python script. Monthly beer production
#
#############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import os

# Configure matplotlib for better display
plt.rcParams['font.size'] = 14

#############################################################################
# Importing a CSV data set
#############################################################################

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the CSV file (in the same directory)
csv_file = os.path.join(current_dir, "monthly-beer-production-in-austr.csv")

# Read the CSV file, parsing the Month column as datetime
data = pd.read_csv(csv_file, parse_dates=['Month'], infer_datetime_format=True)

#############################################################################
# Plotting
#############################################################################

D = data['Month'].values
beer = data['Monthly beer production'].values

plt.figure(figsize=(10, 6))
plt.plot(D, beer)
plt.xlabel('Year')
plt.ylabel('Beer production in Australia in liters')
plt.tight_layout()

#############################################################################
# Analyze ACF
#############################################################################

# Take a subset of the beer data
beer_subset = beer[:200]
D_subset = D[:200]

# Detrend the data
dbeer = signal.detrend(beer_subset)

# Compute ACF (autocorrelation function)
acf_values, confidence_interval = acf(dbeer, nlags=100, alpha=0.05)
lags = np.arange(len(acf_values))

# Plot detrended beer production
plt.figure(figsize=(10, 6))
plt.plot(D_subset, dbeer)
plt.xlabel('Year')
plt.ylabel('Beer production in Australia in liters (detrended)')
plt.tight_layout()

# Plot ACF
plt.figure(figsize=(10, 6))
plt.plot(lags, acf_values)
plt.scatter(lags, acf_values, s=50, zorder=5)
plt.xlabel('Lags')
plt.ylabel('ACF')
plt.tight_layout()

# Display all plots at once
plt.show()