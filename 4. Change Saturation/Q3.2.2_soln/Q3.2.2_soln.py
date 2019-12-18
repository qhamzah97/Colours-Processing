import numpy as np
import cv2
import math


img = cv2.imread('Lena.jpeg',1)
pi = math.pi

S = float(input("input a value to change Saturation of an image. Input value should be between -1 to 1: \n"))
while S < -1.0 or S> 1.0:
    print("invalid input")
    S = float(input("input value should be between -1 to 1: \n"))
print ("The Saturation will be shifted by:  " )  
print(S)

def rgb_to_hsi(img,A):

    img = (img.astype(float)/255)
    RGB_img  = np.zeros((img.shape[0],img.shape[1],3), float)
    HSI_img  = np.zeros((img.shape[0],img.shape[1],3), float)
    r = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    g = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    b = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    theta = np.zeros([img.shape[0],img.shape[1],], dtype = float)
    c_min = np.zeros([img.shape[0],img.shape[1],], dtype = float)
    num = np.zeros([img.shape[0],img.shape[1],], dtype = float)
    den = np.zeros([img.shape[0],img.shape[1],], dtype = float)
    H = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    S = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    I = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    S2 = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r[i,j] = img[i,j][0]
            g[i,j] = img[i,j][1]
            b[i,j] = img[i,j][2]
            num[i,j] = 0.5*((r[i,j]-g[i,j])+(r[i,j]-b[i,j]))
            den[i,j] = math.sqrt(((r[i,j]-g[i,j])**2)+((r[i,j]-b[i,j])*(g[i,j]-b[i,j])))
            theta[i,j] = num[i,j]/den[i,j]
            theta[i,j] = math.acos(theta[i,j])
            c_min[i,j] = min(r[i,j],g[i,j],b[i,j])
            S[i,j] = 1-((3/(r[i,j]+g[i,j]+b[i,j]))*c_min[i,j])
            I[i,j] = (r[i,j]+g[i,j]+b[i,j])/3
            if b[i,j] > g[i,j]:
                H[i,j] =  (2*pi - theta[i,j])
            else:
                 H[i,j] = theta[i,j]
            S2[i,j] = S[i,j]*A
            if (S2[i,j] >1):
                S2[i,j] = 1
            elif (S2[i,j] < 0):
                S2[i,j] = 0
                
    # stacking the R,G,B layers to see if it outputs the original image 
    RGB_img[...,0] = r
    RGB_img[...,1] = g
    RGB_img[...,2] = b
    # stacking the H,S,I layers to get the overal image output
    HSI_img[...,0] = H*255
    HSI_img[...,1] = S2*255
    HSI_img[...,2] = I*255

    return HSI_img

HSI_img = rgb_to_hsi(img,S)
cv2.imwrite('HSI_Sshifted.jpeg', HSI_img)


def hsi_to_rgb(img):
    img = img.astype(float)/255
    RGB_img  = np.zeros((img.shape[0],img.shape[1],3), float)
    #HSI_img  = np.zeros((img.shape[0],img.shape[1],3), float)
    H = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    S = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    I = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    r = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    g = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    b = np.zeros([img.shape[0],img.shape[1]], dtype = float)
    for i in range(img.shape[0]):
        for j in range (img.shape[1]):
            H[i,j] = img[i,j][0]
            S[i,j] = img[i,j][1]
            I[i,j] = img[i,j][2]
            if 0<= H[i,j] < (2/3*pi):
                b[i,j] = I[i,j]*(1-S[i,j])
                r[i,j] = I[i,j]*(1+ (S[i,j]*math.cos(H[i,j]))/(math.cos((pi/3 - H[i,j]))))
                g[i,j] = (3*I[i,j] - (r[i,j] +b[i,j]))
            elif ((2/3*pi)<= H[i,j] < (4/3*pi)):
                r[i,j] = I[i,j]*(1-S[i,j])
                g[i,j] = I[i,j]*(1+((S[i,j]*math.cos((H[i,j] - (2/3*pi))))/(math.cos(pi/3 - (H[i,j] - (2/3*pi))))))
                b[i,j] = (3*I[i,j] - (r[i,j] +g[i,j]))
            elif ((4/3*pi <= H[i,j] <= 2*pi)):
                g[i,j] = I[i,j]*(1- S[i,j])
                b[i,j] = I[i,j]*(1+((S[i,j]*math.cos((H[i,j] - (4/3*pi))))/(math.cos(pi/3 - (H[i,j] - (4/3*pi))))))
                r[i,j] = (3*I[i,j] - (g[i,j] +b[i,j]))
    RGB_img[...,0] = r*255
    RGB_img[...,1] = g*255
    RGB_img[...,2] = b*255

    return RGB_img

RGB_img = hsi_to_rgb(HSI_img)
cv2.imwrite('RGB_lena_from_HSI_Sshifted.jpeg',RGB_img)
