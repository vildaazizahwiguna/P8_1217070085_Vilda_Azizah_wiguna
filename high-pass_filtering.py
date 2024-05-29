# Vilda Azizah Wiguna (1217070085)

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('kucing_helm.webp',0)

#--menerapkan algoritma high-pass filtering:
# laplacian
laplacian = cv2.Laplacian(img,cv2.CV_64F)

# sobel dengan ukuran kernel 5
sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

#perbesar ukuran hasil plotting
plt.rcParams["figure.figsize"] = (20,20)

#menampilkan hasil filter
plt.subplot(2,2,1), plt.imshow(img,cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2), plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,3), plt.imshow(sobelx,cmap='gray')
plt.title('Sobel x'), plt.xticks([]), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,4), plt.imshow(sobely,cmap='gray')
plt.title('Sobel y'), plt.xticks([]), plt.xticks([]), plt.yticks([])
plt.show()