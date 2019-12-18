import numpy as np 
import cv2 

in_img = cv2.imread('lena.jpeg',1)

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
