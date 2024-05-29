# Vilda Azizah Wiguna (1217070085)

import cv2
import numpy as np
from matplotlib import pyplot as plt

# Membaca gambar dalam mode grayscale
image_path = r"kucing_helm.webp"
image = cv2.imread(image_path, 0)

# Hitung Discrete Fourier Transform (DFT)
DFT = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)

# Geser DFT sehingga komponen frekuensi nol berada di tengah
shift = np.fft.fftshift(DFT)
row, col = image.shape
center_row, center_col = row // 2, col // 2

# Buat mask dengan ukuran 60x60 di tengah
mask = np.zeros((row, col, 2), np.uint8)
mask[center_row - 30:center_row + 30, center_col - 30:center_col + 30] = 1

# Terapkan mask ke DFT yang telah digeser
fft_shift = shift * mask

# Geser balik sebelum inverse DFT
fft_ifft_shift = np.fft.ifftshift(fft_shift)

# Inverse DFT untuk mendapatkan kembali citra spasial
image_back = cv2.idft(fft_ifft_shift)

# Hitung magnitude dari hasil inverse DFT
image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

# Visualisasi
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(image, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(image_back, cmap='gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
