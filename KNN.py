import numpy as np
from collections import Counter

class KNN:
    def euclidean(self,X1,X2):
        return np.sqrt((np.array(X2)-np.array(X1))**2)

    def knn_predict(self,training_data,training_labels,test_points,k):
        distances = []
        for i in range(len(training_data)):
            dist = self.euclidean(test_points,training_data[i])
            distances.append((dist,training_labels[i]))
        distances.sort(key=lambda x: x[0][0])
        k_nearest_labels = [labels for _,labels in distances[:k]]
        return Counter(k_nearest_labels).most_common(1)[0][0]

training_data = [[1, 2], [2, 3], [3, 4], [6, 7], [7, 8]]
training_labels = ['A', 'A', 'A', 'B', 'B']
test_point = [4, 5]
k = 3

kn = KNN()
prediction = kn.knn_predict(training_data, training_labels, test_point, k)
print(prediction)