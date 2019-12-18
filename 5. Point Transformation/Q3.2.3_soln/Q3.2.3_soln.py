import numpy as np
import cv2

in_img = cv2.imread('lena.jpeg',1)

C = float(input("Give the scalar multiplier 'C' any real value: "))
print (C)

B = float(input("Give 'B' any real value between 0 to 255: "))
print(B)
while B<0.0 or B>255.0:
    print("invalid input choose another value for B \n")
    B = float(input("Give 'B' any real value between 0 to 255: "))
    print(B)

def apply_point_tfrm(img, B, C):
    image = np.zeros((img.shape[0], img.shape[1], 3),float)
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]

    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            image[x,y,0] = C*r[x,y] + B
            image[x,y,1] = C*g[x,y] + B
            image[x,y,2] = C*b[x,y] + B
            for c in range(0,2):
                if image[x,y,c] > 255:
                    image[x,y,c] = 255
                if image[x,y,c] < 0:
                    image[x,y,c] = 0
       
    return image

imgTransformed = apply_point_tfrm(in_img, B, C)
cv2.imwrite('lena_new.jpg', imgTransformed)