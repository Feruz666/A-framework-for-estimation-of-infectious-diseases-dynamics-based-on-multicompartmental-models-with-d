import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, make_interp_spline


dataset = pd.read_csv("new_cases.csv")

col_list = ["date", "Russia"]
date = dataset['date'].tolist()
russia = dataset['Russia'].tolist()



date_data = []
russia_data = []
for i in range(0, 350):
    date_data.append(date[i])
    russia_data.append(russia[i])


n = 10
date_data = date_data[n:]
russia_data = russia_data[n:]

x = np.arange(len(russia_data))
M_BSpline = make_interp_spline(x, russia_data)
xm_new = np.arange(x[0], x[-1], 0.1)
ym_new = M_BSpline(xm_new)


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(xm_new, ym_new)
ax.tick_params(axis='x', rotation=85)
ax.grid()
plt.show()
