from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = np.array([[1],[4],[5],[0]])
y = np.array([0,1,1,0])
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
alg = KNeighborsClassifier(n_neighbors = 1)
alg.fit(Xtrain,ytrain)
ypred = alg.predict(Xtest)
color1 = np.where(ytrain == 0, "green","red")
color2 = np.where(ypred == 0, "green","red")
plt.scatter(Xtrain , np.zeros(len(Xtrain)), c = color1)
plt.scatter(Xtest, np.zeros(len(Xtest)), c = color2, marker = "x")
plt.grid()
plt.figure()

X = np.array([[1,-1],[4,3],[5,6],[0,0]])
y = np.array([0,1,1,0])
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
alg = KNeighborsClassifier(n_neighbors=1)
alg.fit(Xtrain,ytrain)
ypred = alg.predict(Xtest)
color1 = np.where(ytrain == 0, "green","red")
color2 = np.where(ypred == 0, "green","red")
xcord = Xtrain[:,0]
ycord = Xtrain[:,1]
xcord2 = Xtest[:,0] # Stores the x coordinates
ycord2 = Xtest[:,1]
plt.scatter(xcord,ycord, c = color1)
plt.scatter(xcord2,ycord2, c = color2, marker = "x")
plt.axis("equal")
plt.grid()