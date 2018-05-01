import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

arr_1= np.array([12,23,34,45,43,456])
arr_2 = np.arange(0,20)
arr_3 = np.random.random()
arr_4 = np.random.randint(0,10)
arr_4 = np.random.randint(0,100,30)
arr_5 = np.array([[1,2,3],[3,4,5]])
x = np.array([i for i in range(10)])
y = np.random.randint(200,1000,10)
plt.plot(x,y)
plt.show()
x = np.array([i for i in range(10)])
y = np.random.randint(200,1000,10)
z=np.random.randint(1,50,10)
fig = plt.figure(figsize=(12,6))
ax = Axes3D(fig)
ax.scatter(x,y,z,color='red')
plt.shw()

pd.read_csv('matches.csv')
dataset=pd.read_csv('matches.csv')
dataset.shape
obs_1 = dataset.head()
obs_2 = dataset.tail()
plt.hist(dataset['season'],rwidth=0.8)

sns.countplot(x=dataset['season'],data= dataset)