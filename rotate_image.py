#!/usr/bin/python3.8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.transforms as mtransforms
from scipy.ndimage import rotate
import os
import sys
from PIL import Image


########################################################

def get_image(fn = None):
    
    if fn == None:
        n = np.random.choice(1000)
        t = np.random.choice(["Arborio", "basmati", "Ipsala", "Jasmine", "Karacadag"])
        print(t)
        fn = "Rice_Image_Dataset/%s/%s (%d).jpg" % ( t, t, n)

    rice = image.imread(fn)
    
    return rice
    
def get_angle(img):
        
    y,x = np.nonzero(img)
    z   = img[y,x]
    
    x   = x - np.mean(x)
    y   = y - np.mean(y)
    xy  = np.vstack([y,x])

    cov = np.cov(xy,aweights=z)
    evals,evecs = np.linalg.eig(cov)
    
    sort_indices = np.argsort(evals)[::-1]
    x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
    x_v2, y_v2 = evecs[:, sort_indices[1]]    
    
    return 180*np.arctan((x_v1)/(y_v1))/np.pi
    
def do_plot(ax, Z, transform):
    im = ax.imshow(Z, interpolation='none',
                   origin='lower',
                   extent=[-2, 4, -3, 2], clip_on=True)

    trans_data = transform + ax.transData
    im.set_transform(trans_data)

    # display intended extent of the image
    x1, x2, y1, y2 = im.get_extent()
    ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "y--",
            transform=trans_data)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-4, 4)
    
#######################################################

for rice_type in ["Arborio", "basmati", "Ipsala", "Jasmine", "Karacadag"]:
    for rice_img in np.sort(os.listdir("Rice_Image_Dataset/%s/" % rice_type)):
        fn = "Rice_Image_Dataset/%s/%s" % ( rice_type, rice_img )
        fn_= "Rice_Roted_Dataset/%s/%s" % ( rice_type, rice_img )
        print(rice_img)
        rice = get_image(fn)
        rice_= rice.sum(axis=2)
        theta= get_angle(rice_)
        rice_= rotate(rice,theta-135,reshape=False)
        im = Image.fromarray(rice_)
        im.save(fn_)
        
exit()
     
    
#######################################################

rice = get_image()

rice_= rice.sum(axis=2)

plt.figure()
plt.imshow(rice)

theta = get_angle(rice_)

print(theta)

rice_ = rotate(rice,theta-135,reshape=False)

print(rice_.shape)

plt.figure()
plt.imshow(rice_)
plt.show()








exit()

#######################################################

rice_[rice_<rice_.mean()] = 0

y,x = np.nonzero(rice_)
z   = rice_[y,x]

x = x - np.mean(x)
y = y - np.mean(y)
coords = np.vstack([x, y])

cov = np.cov(coords,aweights=z)
evals, evecs = np.linalg.eig(cov)

sort_indices = np.argsort(evals)[::-1]
x_v1, y_v1 = evecs[:, sort_indices[0]]  # Eigenvector with largest eigenvalue
x_v2, y_v2 = evecs[:, sort_indices[1]]

scale = 20

plt.scatter(x,y,c=z)
plt.plot([x_v1*-scale*2, x_v1*scale*2],
         [y_v1*-scale*2, y_v1*scale*2], color='red')
plt.plot([x_v2*-scale, x_v2*scale],
         [y_v2*-scale, y_v2*scale], color='blue')
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
plt.show()


theta = np.arctan((x_v1)/(y_v1))  
rotation_mat = np.matrix([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
transformed_mat = rotation_mat * coords
# plot the transformed blob
x_transformed, y_transformed = transformed_mat.A
plt.scatter(x_transformed, y_transformed, c=z)
plt.axis('equal')
plt.gca().invert_yaxis()  # Match the image system with origin at top left
plt.show()

