import numpy as np
import cv2
from kMeans12 import KMeans

imageName = 'bluebutterfly'
imageType = '.jpg'
imagePath = 'images/' + imageName + imageType

img = cv2.imread(imagePath)
print('img dimesnions:', img.shape)

img2 = img.reshape((-1,3))
print('img2 dimensions:', img2.shape)

img2 = np.float32(img2)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#Clusters
k = 6
attempts = 10

#ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
#label = label.flatten()

k = KMeans(K=k, max_iters=10, plot_steps=False)
y_pred, centroid_locations = k.predict(img2)
center = centroid_locations
y_pred = [int(i) for i in y_pred]
label = y_pred
knnType = "Chris"

print("Center:", center)
#print("Label:", label)

center = np.uint8(center)

res = center[label]
print(res)
res2 = res.reshape((img.shape))
cv2.imwrite(imageName+"/"+imageName + ' - ' + knnType + imageType , res2)