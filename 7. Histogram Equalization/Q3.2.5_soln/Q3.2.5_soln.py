import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#def histogram_equalize(img):
#    image = np.zeros((img.shape[0], img.shape[1], 3),float)

#    r = img[:,:,0]
#    g = img[:,:,1]
#    b = img[:,:,2]

#    intensity_array = np.zeros((256, 3),float)
    
#    for i in range(img.shape[0]):
#        for j in range(img.shape[1]):
#            for c in range(0,2):
#                intensity = img[i,j,c]
#                intensity_array[intensity,c] = intensity_array[intensity,c] + 1


#    MN = np.zeros(3)
#    for i in range(1, 256):
#        for c in range(0,2):
#            MN[c] = MN[c] + intensity_array[i,c]
    
#    probability_array = np.zeros((256, 3),float)
#    for c in range(0,2):
#        for i in range(0, 256):
#            probability_array[i,c] = intensity_array[i,c]/MN[c]
#    #print(probability_array)
#    #print(probability_array.shape)

#    CDF = np.zeros(3)
#    CDF_array = np.zeros((256, 3),float)
#    for i in range(1, 256):
#        for c in range(0,2):
#            CDF[c] = CDF[c] + probability_array[i,c]
#            CDF_array[i,c] = CDF[c]
#    #print(CDF_array)
    
#    final_array = np.zeros((256, 3),float)
#    final_array = (CDF_array * 255)
#    for i in range (1,256):
#        for c in range(0,2):
#            final_array[i,c] = math.ceil(final_array[i,c])
#            if(final_array[i,c] > 255):
#                final_array[i,c] = 255
#    #print(final_array)

#    for i in range(img.shape[0]):
#        for j in range(img.shape[1]):
#            for value in range(0, 255):
#                for c in range(0,2):
#                    if (img[i,j,c] == value):
#                        image[i,j,c] = final_array[value,c]
#                        break
#    print(image)
#    return image

def histogram_equalize(img):
    image = np.asarray(img)

    intensity_array = np.zeros(256)
    rows, columns = image.shape[:2]
    for i in range(0, rows):
        for j in range(0, columns):
            intensity = image[i,j]
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

in_img = cv2.imread('lena.jpeg',1)
out_image = np.zeros((in_img.shape[0], in_img.shape[1], 3),float)
r = in_img[:,:,0]
g = in_img[:,:,1]
b = in_img[:,:,2]

out_r = histogram_equalize(r)
cv2.imwrite('lena_r.jpeg', out_r)

out_g = histogram_equalize(g)
cv2.imwrite('lena_g.jpeg', out_g)

out_b = histogram_equalize(b)
cv2.imwrite('lena_b.jpeg', out_b)

out_image[:,:,0] = r
out_image[:,:,1] = g
out_image[:,:,2] = b
cv2.imwrite('lena_final.jpeg', out_image)
