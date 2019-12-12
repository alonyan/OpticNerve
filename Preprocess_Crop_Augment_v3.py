# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:07:49 2019

@author: EJMorales
AOY - added cropping
"""
import glob
import numpy as np
from PIL import ImageOps
from PIL import Image
import keras.preprocessing as kp #import image
import joblib
from PIL import ImageEnhance
from skimage.util import invert
from skimage.color import rgb2gray
#import matplotlib.pyplot as plt
 
############################################
############################################
# Load and process each image
# Images are converted into arrays and labeled as 0 or 1
# 0 : excluded image
# 1 : included image
images = []
imageNames = []
labels = []
size = 128, 128

#crpSize = 1024


crpSize = 64

#imageFolder = 'GROUP_1.5//*jpg
#%%   
#label = 0

def hist_match_tonorm(source):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches a normal distribution

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """
    from scipy import stats

    oldshape = source.shape
    source = source.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    t_values = np.arange(0,255)
    t_quantiles = stats.norm.cdf(t_values, loc=128, scale=40)
    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape) 
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

#%%
def loadAndLabel_Rotate(imageFolder, images, imageNames, labels, label):
    # Excluded images
    total = len(glob.glob(imageFolder))
    for i, img_path in enumerate(glob.glob(imageFolder)):
        #if i > 1:
            #break
        print("{} : {}".format(i, total))
        imageNames.append(img_path)
        
        # Load image
        #img = image.load_img(img)
        img_og = Image.open(img_path)
        
        #first resize - 2056/128=16
        newSize = np.array(img_og.size)//16

        img_og = img_og.resize(newSize.tolist())
        # get pixel array
        pix = np.array(img_og)

             
        # the green channel seems like it's the most informative. We can also do rgb2gray or even calculate 
        
        #First, crop away padding, This would automatically remove what is obviously BG
        nonZero = np.argwhere(pix[:,:,1]>5);
        edgesize = 4
        pix = pix[np.min(nonZero[:,0])+edgesize:np.max(nonZero[:,0])-edgesize, np.min(nonZero[:,1])+edgesize:np.max(nonZero[:,1])-edgesize,:]
         
        # fourier low-pass-filter for illumination correction
        p,s =  periodic_smooth_decomp(invert(rgb2gray(pix)))
        img_fft = np.fft.fft2(p)      
        img_fft = np.fft.fftshift(img_fft)
        imsize = pix.shape[:2]
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
        #normalize
        img_ridges = (img_ridges-np.min(img_ridges))/(np.max(img_ridges)-np.min(img_ridges))
        img_smooth = (img_smooth-np.min(img_smooth))/(np.max(img_smooth)-np.min(img_smooth))

        #mask and create local density of ridges
        mask = (img_ridges>trithresh(img_ridges))
        mask_fft = np.fft.fftshift(np.fft.fft2(mask))

        kernel = np.exp(-0.2*radialDist**2)
        mask = np.real(np.fft.ifft2(np.fft.ifftshift(mask_fft*kernel)))

        # create weight matrix, weigh "ridgeness", area intensity
        weightMat = (1-img_smooth)*mask
        
        # find center of mass
        crpcenter = np.unravel_index(np.argmax(weightMat),weightMat.shape[0:2])

                
        #img_ridges = filters.laplace(filters.gaussian(np.invert(pix[:,:,1]),sigma=20))
        
        x0 = np.max((np.max((crpcenter[0]-crpSize,0))-np.max((crpcenter[0]+crpSize,imsize[0]))+imsize[0],0))
        
        x1 = np.min((np.min((crpcenter[0]+crpSize,imsize[0]))-np.min((crpcenter[0]-crpSize,0)),imsize[0]))
        
        y0 = np.max((np.max((crpcenter[1]-crpSize,0))-np.max((crpcenter[1]+crpSize,imsize[1]))+imsize[1],0))
        y1 = np.min((np.min((crpcenter[1]+crpSize,imsize[1]))-np.min((crpcenter[1]-crpSize,0)),imsize[1]))
    
        pix = pix[x0:x1,y0:y1,:]
        
        #pix[:,:,0] = hist_match_tonorm(pix[:,:,0])
        #pix[:,:,1] = hist_match_tonorm(pix[:,:,1])
        #pix[:,:,2] = hist_match_tonorm(pix[:,:,2])
        
        
        img_og = Image.fromarray(pix.astype('uint8'), 'RGB')

        img_og = img_og.resize(size)    

        #add black border
        '''
        for img in len(img_path):
            tempImg = Image.new("RGB", (128,128))   ## luckily, this is already black!
            newImg = img
            tempImg.paste(newImg.resize((110,110)), (9,9))                    
            img = tempImg
        '''
        #img.rotate(random.uniform(-30, 30))                        
        for j in range(0,8):
            img = img_og
            if j == 0:
                pass
            if j == 1:               
                img = img.rotate(45)
            if j == 2:                
                img = img.rotate(180)
            if j == 3:                
                img = img.rotate(-45)
            if j == 4:
                img = ImageOps.mirror(img)  
            if j == 5:
                # Re-open the image, apply a 10 percent crop
                # Then re-size
                img = Image.open(img_path)
                width = img.size[0]
                height = img.size[1]
            
                img = img.crop((round(width*.10,0),
                                round(height*.10,0),
                                width-round(width*.10,0),
                                height-round(height*.10,0)))
                img = img.resize(size)
            if j == 6:
                     tempImg = Image.new("RGB", (128,128))   ## luckily, this is already black!
                     tempImg.paste(img.resize((110,110)), (9,9))                    
                     img = tempImg
                #img.save("Test_Crop_10_Percent__128x128.jpg")
            if j == 7:
               enhancer = ImageEnhance.Contrast(img)
               img = enhancer.enhance(1.5)
            
            
            
            # If we're dealing with B&W make it 3 channel image duplicate
            if kp.image.img_to_array(img).shape[2] == 1:
                img = np.array(img)
                img = np.stack((img,)*3, axis=-1)
                
            
            # Convert image to a numpy array
            image_array = kp.image.img_to_array(img)
            # Add to list of images
            images.append(image_array)
        
            # Add the corresponding label
            labels.append(label)


#%%
loadAndLabel_Rotate(dataFolder+"GROUP_1.5/*jpg", images, imageNames, labels, 1)
loadAndLabel_Rotate(dataFolder+'GROUP_3.0/*jpg', images, imageNames, labels, 2)
loadAndLabel_Rotate(dataFolder+'GROUP_4.0/*jpg', images, imageNames, labels, 3)
loadAndLabel_Rotate(dataFolder+'GROUP_5.0/*jpg', images, imageNames, labels, 4)
#%%
images_data_raw = np.array(images)
images_data_labels = np.array(labels)
images_data_files_names = np.array(imageNames)


# Save features
# Save the array of extracted features to a file
joblib.dump(images_data_raw, "images_new_4_groups_last_.dat")

# Save the matching array of expected values to a file
joblib.dump(images_data_labels, "images_new_4_groups_labels_last.dat")

# Save the file locations
joblib.dump(images_data_files_names, "images_new_4_groups_image_names_last.dat")



############################################