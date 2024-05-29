# Vilda Azizah Wiguna (1217070085)

#memanggil modul yang diperlukan
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

#BGR
img = cv2.imread('kucing_helm.webp')

#RGB
kucing = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#tampilkan gambar awal tanpa filter
plt.imshow(img)
plt.show()

#membuat filter : matriks berukuran 5x5
kernel = np.ones((5,5),np.float32)/25
print(kernel)

#--cara lain membuat kernel berukuran 3x3
kernel = np.matrix([
    [1,1,1],
    [1,2,1],
    [1,1,1]
])/25
print(kernel)

#lakukan filtering
cat_filter = cv2.filter2D(img, -1, kernel)

#tampilkan gambar awal tanpa filter
plt.imshow(cat_filter)
plt.show()

#perbesaran ukuran hasil plotting jika diperlakukan
plt.rcParams["figure.figsize"] = (15,15)

#plot pertama, gambar asli
plt.subplot(121),plt.imshow(kucing),plt.title('Original')
plt.xticks([]),plt.yticks([])

#kedua, hasil filter
plt.subplot(122),plt.imshow(cat_filter)
plt.title('averaging')
plt.xticks([]),plt.yticks([])

#plot
plt.show()

cat_blur = cv2.blur(img,(5,5))

#tampilkan gambar awal tanpa filter 
plt.imshow(cat_blur)
plt.show()