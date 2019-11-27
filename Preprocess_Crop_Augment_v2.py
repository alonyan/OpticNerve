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

crpSize = 900
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
        # get pixel array
        pix = np.array(img_og)

             
        # the green channel seems like it's the most informative. We can also do rgb2gray or even calculate 
        
        #First, crop away padding, Also remove edges here
        nonZero = np.argwhere(pix[:,:,1]>5);
        pix = pix[np.min(nonZero[:,0])+400:np.max(nonZero[:,0])-400, np.min(nonZero[:,1])+400:np.max(nonZero[:,1])-400,:]
               
        # fourier low-pass-filter for illumination correction
        img_fft = np.fft.fft2(pix[:,:,1])

        
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
        # We'll do a DoG using ffts
        #Low freq:
        kernel = np.exp(-1*radialDist**2)
        kernel1 = np.exp(-.001*radialDist**2)
        ff_corr_pix = np.abs(np.fft.ifft2(np.fft.ifftshift(img_fft*(kernel-kernel1))))
 
        # green seems the most informative. Find index of maximum
        indmax = np.unravel_index(np.argmax(ff_corr_pix),ff_corr_pix.shape[0:2])
        
        crpcenter = np.array(indmax)
        
        
        #img_ridges = filters.laplace(filters.gaussian(np.invert(pix[:,:,1]),sigma=20))
        crpcenter = crpcenter.astype('int16')
        pix = pix[np.max((crpcenter[0]-crpSize,0)):np.min((crpcenter[0]+crpSize,imsize[0])),np.max((crpcenter[1]-crpSize,0)):np.min((crpcenter[1]+crpSize,imsize[1])),:]
        
        pix[:,:,0] = hist_match_tonorm(pix[:,:,0])
        pix[:,:,1] = hist_match_tonorm(pix[:,:,1])
        pix[:,:,2] = hist_match_tonorm(pix[:,:,2])
        
        
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
dataFolder = "/RazorScopeData/RazorScopeImages/Diana/Project- optic disc quality/"
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