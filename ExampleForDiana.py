#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 16:14:46 2019

@author: alon
"""

#%%
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.util import invert

dataFolder = "/RazorScopeData/RazorScopeImages/Diana/Project- optic disc quality/"
#%%
def periodic_smooth_decomp(I: np.ndarray) -> (np.ndarray, np.ndarray):
    '''Performs periodic-smooth image decomposition
    Parameters
    ----------
    I : np.ndarray
        [M, N] image. will be coerced to a float.
    Returns
    -------
    P : np.ndarray
        [M, N] image, float. periodic portion.
    S : np.ndarray
        [M, N] image, float. smooth portion.
        
        Code from: https://github.com/jacobkimmel/ps_decomp
    '''
    u = I.astype(np.float64)
    v = u2v(u)
    v_fft = np.fft.fftn(v)
    s = v2s(v_fft)
    s_i = np.fft.ifftn(s)
    s_f = np.real(s_i)
    p = u - s_f # u = p + s
    return p, s_f

def u2v(u: np.ndarray) -> np.ndarray:
    '''Converts the image `u` into the image `v`
    Parameters
    ----------
    u : np.ndarray
        [M, N] image
    Returns
    -------
    v : np.ndarray
        [M, N] image, zeroed expect for the outermost rows and cols
    '''
    v = np.zeros(u.shape, dtype=np.float64)

    v[0, :] = np.subtract(u[-1, :], u[0,  :], dtype=np.float64)
    v[-1,:] = np.subtract(u[0,  :], u[-1, :], dtype=np.float64)

    v[:,  0] += np.subtract(u[:, -1], u[:,  0], dtype=np.float64)
    v[:, -1] += np.subtract(u[:,  0], u[:, -1], dtype=np.float64)
    return v

def v2s(v_hat: np.ndarray) -> np.ndarray:
    '''Computes the maximally smooth component of `u`, `s` from `v`
    s[q, r] = v[q, r] / (2*np.cos( (2*np.pi*q)/M )
        + 2*np.cos( (2*np.pi*r)/N ) - 4)
    Parameters
    ----------
    v_hat : np.ndarray
        [M, N] DFT of v
    '''
    M, N = v_hat.shape

    q = np.arange(M).reshape(M, 1).astype(v_hat.dtype)
    r = np.arange(N).reshape(1, N).astype(v_hat.dtype)

    den = (2*np.cos( np.divide((2*np.pi*q), M) ) \
         + 2*np.cos( np.divide((2*np.pi*r), N) ) - 4)
    s = np.divide(v_hat, den, out=np.zeros_like(v_hat), where=den!=0)
    s[0, 0] = 0
    return s
#%%
def trithresh(pix, nbins=256):
    imhist, edges = np.histogram(pix[:],nbins)
    centers = (edges[1:]+edges[:-1])/2
    
    a = centers[np.argmax(np.cumsum(imhist)/np.sum(imhist)>0.9999)] #brightest
    b = centers[np.argmax(imhist)] #most probable
    h = np.max(imhist) #response at most probable
    
    m = h/(b-a)
    
    x1=np.arange(0,a-b, 0.1)
    y1=np.interp(x1+b,centers,imhist)
    
    L = (m**2+1)*((y1-h)*(1/(m**2-1))-x1*m/(m**2-1))**2 #Distance between line m*x+b and curve y(x) maths!
    
    triThresh = b+x1[np.argmax(L)]
    return triThresh

#%% Load, resize, remove borders
Files = glob.glob(dataFolder+"GROUP_1.5/*jpg")
img_path = Files[82]

crpSize = 64

img_og = Image.open(img_path)

newSize = np.array(img_og.size)//16

img_og = img_og.resize(newSize.tolist())
# get pixel array
pix = np.array(img_og)
plt.imshow(pix)

#First, crop away padding, This would automatically remove what is obviously BG
nonZero = np.argwhere(pix[:,:,1]>5);
edgesize = 3
pix = pix[np.min(nonZero[:,0])+edgesize:np.max(nonZero[:,0])-edgesize, np.min(nonZero[:,1])+edgesize:np.max(nonZero[:,1])-edgesize,:]
     
#%%
rgbimg = rgb2gray(pix)

p,s =  periodic_smooth_decomp(invert(rgbimg))


img_fft = np.fft.fft2(p);

img_fft = np.fft.fftshift(img_fft)


imsize = pix.shape[:2];
#dimensions in x and y
y = imsize[0]
x = imsize[1]
#position of center
centY = np.ceil(y/2)
centX = np.ceil(x/2)
#create the grid
yy,xx = np.indices((y,x))
radialDist = np.sqrt((xx-centX)**2 + (yy - centY)**2)

# make LoG image for ridges
kernel = np.exp(-0.001*radialDist**2)
img_ridges = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel*(radialDist**2))))

#make smooth image for intensity weight
kernel = np.exp(-0.1*radialDist**2)
img_smooth = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft*kernel)))

#img_smooth = filters.gaussian(np.invert(pix[:,:,1]),sigma=20)
#img_ridges = filters.laplace(img_smooth)
#a = pix[:,:,1]*(img_ridges>filters.threshold_triangle(img_ridges))

img_ridges = (img_ridges-np.min(img_ridges))/(np.max(img_ridges)-np.min(img_ridges))
img_smooth = (img_smooth-np.min(img_smooth))/(np.max(img_smooth)-np.min(img_smooth))


mask = img_ridges>trithresh(img_ridges);
p,s =  periodic_smooth_decomp(mask)

mask_fft = np.fft.fftshift(np.fft.fft2(p))

kernel = np.exp(-0.2*radialDist**2)
mask = np.real(np.fft.ifft2(np.fft.ifftshift(mask_fft*kernel)))

weightMat = (1-img_smooth)*mask

plt.imshow(weightMat)

#%%
indmax = np.unravel_index(np.argmax(weightMat),weightMat.shape[0:2])

#indmax = np.sum(xx*a)/np.sum(a), np.sum(yy*a)/np.sum(a)
plt.imshow(pix)
plt.scatter(indmax[1],indmax[0])
crpcenter = indmax
#%%
x0 = np.max((np.max((crpcenter[0]-crpSize,0))-np.max((crpcenter[0]+crpSize,imsize[0]))+imsize[0],0))
        
x1 = np.min((np.min((crpcenter[0]+crpSize,imsize[0]))-np.min((crpcenter[0]-crpSize,0)),imsize[0]))
        
y0 = np.max((np.max((crpcenter[1]-crpSize,0))-np.max((crpcenter[1]+crpSize,imsize[1]))+imsize[1],0))
y1 = np.min((np.min((crpcenter[1]+crpSize,imsize[1]))-np.min((crpcenter[1]-crpSize,0)),imsize[1]))
    
pixSmall = pix[x0:x1,y0:y1,:]
        
plt.imshow(pixSmall)