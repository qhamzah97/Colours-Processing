import sys 
import numpy as np
import cv2
from scipy import misc
from scipy.ndimage import convolve
from scipy import signal

def spatial_filter(img,h):
    img = img.astype(float)
    img = signal.convolve2d(img,h, boundary = 'symm', mode ='same')
    return img

in_img = cv2.imread('lena.jpeg',1)
out_image = np.zeros((in_img.shape[0], in_img.shape[1], 3),float)
r = in_img[:,:,0]
g = in_img[:,:,1]
b = in_img[:,:,2]

h = [[-1, -1, -1],
     [-1, 8, -1],
     [-1, -1, -1]]

out_r = spatial_filter(r,h)
cv2.imwrite('lena_r.jpeg', out_r)

out_g = spatial_filter(g,h)
cv2.imwrite('lena_g.jpeg', out_g)

out_b = spatial_filter(b,h)
cv2.imwrite('lena_b.jpeg', out_b)

out_image[:,:,0] = r
out_image[:,:,1] = g
out_image[:,:,2] = b
cv2.imwrite('lena_final.jpeg', out_image)