# Vilda Azizah Wiguna (1217070085)

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('baymax.jpg')

#memanggili fungsi canny edges dengan argumen (citra, nilai_min, nilai_max)
edges = cv2.Canny(img,50,100)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('original'),plt.xticks([]), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('edge image'),plt.xticks([]), plt.xticks([]), plt.yticks([])
plt.show()