import numpy as np

predictions = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\predictions.csv', 
                           delimiter=',',dtype=float)

test_data = np.genfromtxt('C:\\Users\\brett\\source\\repos\\MLProject2\\MLProject2\\netflix\\TestingRatings.txt', 
                           delimiter=',',dtype=int)

truth = test_data[:,2]

delta = truth - predictions


print(delta)

print(np.sum(delta) / truth.size)
print(np.sqrt(np.sum(delta * delta) / truth.size))