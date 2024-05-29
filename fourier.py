# Vilda Azizah Wiguna (1217070085)

import cv2 
import numpy as np 
import matplotlib.pyplot as plt

image = cv2.imread(r"kucing_helm.webp")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
fourier = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
fourier_shift = np.fft.fftshift(fourier)

#hitung fourier transform
magnitude = 20*np.log(cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1]))

#menormalisasi
magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

#display
plt.imshow((magnitude))
plt.show()