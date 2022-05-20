import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('dog.jpg', 0)

f = np.fft.fft2(img)

fshift = np.fft.fftshift(f)

result = 20*np.log(np.abs(fshift))

plt.subplot(121)
plt.imshow(img, cmap='gray')
plt.title('original')
plt.axis('off')

plt.subplot(122)
plt.imshow(result, cmap='gray')
plt.title('result')
plt.axis('off')

plt.show()