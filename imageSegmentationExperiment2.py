import numpy as np
import cv2
from kMeans12 import KMeans
import time
import os

# 'bluebutterfly', 'Chris', 'egg',  'Mt. Fuji', 'peppers', 'rainbowpallete', 'Truman', 'X', 'yellowmouse', 'White House', 'GoogleLogoSmall', 'ChrisSmall'
# '.jpg', '.jpg', '.jpg', '.jpg' ,'.png', '.jpg', '.jpg', '.png', '.jpg', '.jpg', '.jpg', '.jpg'
images = ['bluebutterfly']
imageTypes = ['.jpg']
kValues = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100] #
attempts = 25
folderNameEnding = ' attempts- ' + str(attempts)

def setUpDirectory(name):
    # define the name of the directory to be created
    path = os.getcwd()+'/'+ name + folderNameEnding

    # define the access rights
    #access_rights = 0o755

    try:
        os.mkdir(path) # , access_rights
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)

for imgIdx in range(len(images)):

    imageName = images[imgIdx]
    imageType = imageTypes[imgIdx]
    imagePath = 'images/' + imageName + imageType

    setUpDirectory(imageName)

    img = cv2.imread(imagePath)
    print('img dimesnions:', img.shape)

    img2 = img.reshape((-1,3))
    print('img2 dimensions:', img2.shape)

    img2 = np.float32(img2)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    print()
    for k in kValues:
        print("Testing K="+ str(k) + '\n')

        print("Running Chris!")
        km = KMeans(K=k, max_iters=attempts, plot_steps=False)
        y_pred, centroid_locations, times = km.predict(img2)
        center = centroid_locations
        y_pred = [int(i) for i in y_pred]
        label = y_pred
        knnType = "Chris"

        center = np.uint8(center)
        res = center[label]
        res2 = res.reshape((img.shape))
        print("Written?:", cv2.imwrite(imageName + folderNameEnding + "/" + knnType +  ' - ' + 'K=' + str(k) + imageType , res2))

        print()

        print("Running CV2!")
        beginning_time = time.time()
        ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        label = label.flatten()
        print("Total Time:", (time.time() - beginning_time),'\n')
        knnType = "CV2"

        center = np.uint8(center)
        res = center[label]
        res2 = res.reshape((img.shape))
        cv2.imwrite(imageName + folderNameEnding + "/" + knnType +  ' - ' + 'K=' + str(k) + imageType , res2)