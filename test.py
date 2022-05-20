from sys import flags
import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt

def get_dft(arr:np.ndarray, savePath:str):
    dft_arr = cv.dft(np.float32(arr) , flags = cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_arr)
    spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))

    plt.subplot(),plt.imshow(spectrum, cmap = 'gray'),
    plt.xticks([]), plt.yticks([])
    plt.imsave(savePath, spectrum)
    #plt.savefig(savePath, bbox_inches='tight')
    gray_img=cv.imread(savePath, cv.IMREAD_GRAYSCALE)
    os.remove(savePath)
    cv.imwrite(savePath, gray_img)
    print(gray_img)
    return gray_img
   
def get_idft(arr:np.ndarray , savePath:str):
    idft_shift = np.fft.ifftshift(arr)
    idft_arr = cv.idft(idft_shift)
    spectrum = 20*np.log(cv.magnitude(idft_shift[:,:,0], idft_shift[:,:,1]))
    plt.xticks([]), plt.yticks([])
    plt.imsave(savePath, spectrum)
    back_img=cv.imread(savePath, cv.IMREAD_GRAYSCALE)
    os.remove(savePath)
    cv.imwrite(savePath, back_img)
    return back_img

img = cv.imread("frogg.jpg" , cv.IMREAD_GRAYSCALE)
aa = get_dft(img, input("Enter the save path for dft graph."))
#cv.imshow("hi", aa)
#cv.waitKey(0)
#cv.destroyWindow("hi")
#iaa = get_idft("/home/tdd/11111.jpg", input("Enter the save path for idft graph."))
#cv.imshow("hii", iaa)
#cv.waitKey(0)
#cv.destroyWindow("hii")
