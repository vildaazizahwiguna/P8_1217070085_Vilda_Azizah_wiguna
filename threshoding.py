# Vilda Azizah Wiguna (1217070085)

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread('baymax.jpg',0)

#menghitung threshold dan perhatikan nilai ambang batas, bawah dan atas dari tiap fungsi yang diberikan
ret, thresh1 = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img,127,255,cv2.THRESH_TOZERO_INV)

#menampilkan hasil 
titles = ['GAMBAR_ASLI','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

#menampilkan beberapa gambar sekaligus 
for i in range(6):
    # 3 baris, 2 kolom
    plt.subplot(3,2,i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()