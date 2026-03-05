#############################################################################
#
# TSA L2: Python script. AMZN stock prices
#
#############################################################################

import pandas as pd
import matplotlib.pyplot as plt
import os

# Configure matplotlib for better display
plt.rcParams['font.size'] = 14

#############################################################################
# Importing a CSV data set
#############################################################################

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the CSV file (in the same directory)
csv_file = os.path.join(current_dir, "AMZN.csv")

# Read the CSV file, parsing the Date column as datetime
data = pd.read_csv(csv_file, parse_dates=['Date'])

#############################################################################
# Plotting
#############################################################################

dates = data['Date']
high = data['High']

plt.figure(figsize=(10, 6))
plt.plot(dates, high)
plt.xlabel('Year')
plt.ylabel('AMZN stock price in USD, high')
plt.tight_layout()
plt.show()