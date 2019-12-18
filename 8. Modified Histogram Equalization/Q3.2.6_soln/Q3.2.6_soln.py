import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

in_img = cv2.imread('lena.jpeg',1)

def histogram_equalize(img):
    image = np.asarray(img)
    #img = image.astype(unsigned char)
    intensity_array = np.zeros(256).astype(float)
    rows, columns = image.shape[:2]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            intensity = image[i,j]
            intensity = math.ceil(intensity)
            img[i,j] = intensity
            intensity_array[intensity] = intensity_array[intensity] + 1


    MN = 0
    for i in range(1, 256):
        MN = MN + intensity_array[i]
    
    probability_array = intensity_array/MN

    CDF = 0
    CDF_array = np.zeros(256)
    for i in range(1, 256):
        CDF = CDF + probability_array[i]
        CDF_array[i] = CDF
    
    final_array = np.zeros(256)
    final_array = (CDF_array * 255)
    for i in range (1,256):
        final_array[i] = math.ceil(final_array[i])
        if(final_array[i] > 255):
            final_array[i] = 255

    new_image = np.zeros(img.shape)
    for i in range(0, rows):
        for j in range(0, columns):
            for value in range(0, 255):
                if (image[i,j] == value):
                    new_image[i,j] = final_array[value]
                    break
    return new_image

def rgb_to_ycbcr(img, A):
    image = np.zeros((img.shape[0], img.shape[1], 3),float)

    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = [r[i,j], g[i,j], b[i,j]]
            image[i,j,0] = np.matmul(A[0], c) + 0       #Y'
            image[i,j,1] = np.matmul(A[1], c) + 128     #Cb
            image[i,j,2] = np.matmul(A[2], c) + 128     #Cr

    image[:,:,0] = histogram_equalize(image[:,:,0])

    return image

def ycbr_to_rgb(img, B):
    image = np.zeros((img.shape[0], img.shape[1], 3),float)

    Y = img[:,:,0]
    Cb = img[:,:,1]
    Cr = img[:,:,2]
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            c = [Y[i,j]-0, Cb[i,j]-128, Cr[i,j]-128]
            image[i,j,0] = np.matmul(B[0], c)       #r
            image[i,j,1] = np.matmul(B[1], c)       #g
            image[i,j,2] = np.matmul(B[2], c)       #b

    return image


A = [[0.299, 0.587, 0.114],
     [-0.1687, -0.3313, 0.5],
     [0.5, -0.4187, -0.0813]]
ycbr_img = rgb_to_ycbcr(in_img, A)
cv2.imwrite("lena_ycbcr.jpeg", ycbr_img)

B = [[1, 0, 1.402],
     [1, -0.34414, -0.71414],
     [1, 1.772, 0]]
rgb_img = ycbr_to_rgb(ycbr_img, B)
cv2.imwrite("lena_rgb.jpeg", rgb_img)

