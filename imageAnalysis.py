import numpy as np
import matplotlib as plt
from PIL import Image
img = Image.open('peppers.png')
A=np.asarray(img, dtype=np.float32)
A = A[:, :, :3] # Remove the alpha channel
img.show()

def showSegmentationResults(clusterCenters, classification, nbRows, nbCols):
    segmentation = clusterCenters[classification]
    imSeg = segmentation.reshape(nbRows, nbCols, 3).astype(np.uint8)
    plt.figure()
    plt.imshow(Image.fromarray(imSeg))
    plt.draw()
