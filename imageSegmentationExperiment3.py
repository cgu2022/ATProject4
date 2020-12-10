import numpy as np
import cv2
from kMeans12 import KMeans
import time
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 'bluebutterfly', 'Chris', 'egg',  'Mt. Fuji', 'peppers', 'rainbowpallete', 'Truman', 'X', 'yellowmouse', 'White House', 'GoogleLogoSmall', 'ChrisSmall'
# '.jpg', '.jpg', '.jpg', '.jpg' ,'.png', '.jpg', '.jpg', '.png', '.jpg', '.jpg', '.jpg', '.jpg'
images = ['bluebutterflysmall']
imageTypes = ['.jpg']
kValues = list(range(2, 10)) + [50, 100] # 4, 5, 6, 7, 8, 9, 10, 20,
attempts = 100
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

    sseList = []
    timesListAvg = []
    timesListSTD = []
    totalTimesList = []
    totalTimesListCV2 = []
    passes = []

    print()
    for k in kValues:
        print("Testing K="+ str(k) + '\n')

        print("Running Chris!")
        km = KMeans(K=k, max_iters=attempts, plot_steps=False)
        y_pred, centroid_locations, times, sse = km.predict(img2)
        sseList.append(sse)
        timesListAvg.append(np.mean(times))
        timesListSTD.append(np.std(times))
        totalTimesList.append(sum(times))
        passes.append(len(times))
        center = centroid_locations
        y_pred = [int(i) for i in y_pred]
        label = y_pred
        knnType = "Chris"

        center = np.uint8(center)
        res = center[label]
        res2 = res.reshape((img.shape))
        print("Written?:", cv2.imwrite(imageName + folderNameEnding + "/" + knnType +  ' - ' + 'K=' + str(k) + imageType , res2))
        print("SSE:", sse)

        print()

        print("Running CV2!")
        beginning_time = time.time()
        ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
        label = label.flatten()
        totalTimesListCV2.append((time.time() - beginning_time))
        print("Total Time:", (time.time() - beginning_time),'\n')
        knnType = "CV2"

        center = np.uint8(center)
        res = center[label]
        res2 = res.reshape((img.shape))
        cv2.imwrite(imageName + folderNameEnding + "/" + knnType +  ' - ' + 'K=' + str(k) + imageType , res2)

    fig = plt.figure()
    plt.plot(kValues, sseList, label = 'SSE Curve', \
            color = 'darkorange', lw = 1, marker = 'o', markersize = 3)

    plt.title("SSE plotted across K-values")
    plt.xlabel('K Parameter')
    plt.ylabel("SSE")
    plt.legend(loc="best")

    # plt.show()
    fig.savefig(imageName + folderNameEnding + "/" + 'SSE Curve Graph.png')


    fig = plt.figure()
    plt.plot(kValues, timesListAvg, label = 'Avg Times Curve', \
            color = 'red', lw = 1, marker = 'o', markersize = 3)

    STDlowerbound = []
    STDupperbound = []
    for i in range(0,len(timesListAvg)):
        STDlowerbound.append(timesListAvg[i]-timesListSTD[i])
        STDupperbound.append(timesListAvg[i]+timesListSTD[i])

    plt.fill_between(kValues, STDlowerbound, \
                  STDupperbound, alpha = 0.2, \
                  color = 'red', lw = 1)

    plt.title("Avg Time plotted across K-values")
    plt.xlabel('K Parameter')
    plt.ylabel("Time (seconds)")
    plt.legend(loc="best")

    # plt.show()
    fig.savefig(imageName + folderNameEnding + "/" + 'Times Curve Graph.png')


    fig = plt.figure()
    plt.plot(kValues, totalTimesList, label = 'Chris Total Time', \
            color = 'navy', lw = 1, marker = 'o', markersize = 3)

    plt.plot(kValues, totalTimesListCV2, label = 'CV2 Total Time', \
            color = 'violet', lw = 1, marker = 'o', markersize = 3)

    plt.title("Total Time plotted across K-values")
    plt.xlabel('K Parameter')
    plt.ylabel("Total Time (seconds)")
    plt.legend(loc="best")

    # plt.show()
    fig.savefig(imageName + folderNameEnding + "/" + 'Total Times Curve Graph.png')


    fig = plt.figure()
    plt.plot(kValues, passes, label = 'Passes', \
            color = 'red', lw = 1, marker = 'o', markersize = 3)

    plt.title("# of Passes plotted across K-values")
    plt.xlabel('K Parameter')
    plt.ylabel("# of Passes")
    plt.legend(loc="best")

    # plt.show()
    fig.savefig(imageName + folderNameEnding + "/" + 'Passes Curve Graph.png')