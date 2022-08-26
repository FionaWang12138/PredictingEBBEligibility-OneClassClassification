import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt

df = pd.read_csv('pairs.csv')
df.rename(columns = {'0':'Features', '1':'Importance'}, inplace = True)
df = df.sort_values(by='Importance', ascending=False)
plt.rcParams['figure.figsize'] = [200,100]
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 100
colors = ['red' if c < 0 else 'blue' for c in df['Importance']]
plt.bar(x=df['Features'], height=df['Importance'],color=colors)
#df.plot.barh(x='Features',color=colors)
plt.title('Feature importances obtained from coefficients')
plt.tick_params(axis='x', which='major', labelsize=100)
plt.xticks(rotation='vertical')
plt.grid()
plt.show()