# Vilda Azizah Wiguna (1217070085)

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar
img = cv2.imread('baymax.jpg')

# Memeriksa apakah gambar berhasil dibaca
if img is None:
    print("Gambar tidak ditemukan atau path salah.")
else:
    # Menghaluskan tepi objek pada citra
    img_blur = cv2.medianBlur(img, 5)

    # Mengonversi gambar ke grayscale
    img_gray = cv2.cvtColor(img_blur, cv2.COLOR_BGR2GRAY)

    # Menerapkan Gaussian filtering sebelum Otsu's thresholding
    blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Otsu's thresholding
    ret, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Plotting gambar asli dan hasil thresholding
    plt.figure(figsize=(10, 5))

    # Menampilkan gambar asli
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Mengonversi BGR ke RGB untuk matplotlib
    plt.title('Gambar Asli')
    plt.axis('off')

    # Menampilkan hasil Otsu's thresholding
    plt.subplot(1, 2, 2)
    plt.imshow(th3, cmap='gray')
    plt.title("Otsu's Thresholding")
    plt.axis('off')

    plt.show()
