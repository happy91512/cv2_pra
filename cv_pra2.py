from operator import ilshift
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
    plt.imsave(savePath, spectrum)
    #plt.savefig(savePath, bbox_inches='tight')
    fre_img=cv.imread(savePath, cv.IMREAD_GRAYSCALE) 
    os.remove(savePath)
    cv.imwrite(savePath, fre_img)
    return dft_shift
   
def get_idft(arr:np.ndarray , savePath:str):
    arr_ishift = np.fft.ifftshift(arr)
    idft_arr = cv.idft(arr_ishift)
    idft_img = cv.magnitude (idft_arr[:,:,0],idft_arr[:,:,1])
    plt.subplot()
    img=plt.imshow(idft_img, cmap = 'gray')
    plt.imsave(savePath, idft_img)
    gray_img=cv.imread(savePath, cv.IMREAD_GRAYSCALE) 
    os.remove(savePath)
    cv.imwrite(savePath, gray_img)
    return gray_img

img = cv.imread("frogg.jpg", cv.IMREAD_GRAYSCALE)
saveP = input("Enter the save path for dft graph:")
aa = get_dft(img, saveP)
f_img =cv.imread(saveP, cv.IMREAD_UNCHANGED)
cv.imshow("hi", f_img) #Show dft img.
cv.waitKey(0)
cv.destroyWindow("hi")
iaa = get_idft(aa, input("Enter the save path for idft graph:"))
cv.imshow("hii", iaa) #Show idft img.
cv.waitKey(0)
cv.destroyWindow("hii") 

