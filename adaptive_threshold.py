# Vilda Azizah Wiguna (1217070085)

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread('baymax.jpg')

# Menghaluskan tepi objek pada citra
img_blur = cv2.medianBlur(img, 5)

# Mengonversi gambar ke grayscale
img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

# Binary threshold
ret, th1 = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive threshold dengan mean
th2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Adaptive threshold dengan gaussian
th3 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Plotting
titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

# Menampilkan hasil
for i in range(4):
    plt.subplot(2, 2, i + 1)
    if i == 0:
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # Konversi gambar asli ke RGB untuk plotting
    else:
        plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()