# %% 
import numpy as np 
import os
import matplotlib.pyplot as plt 
from PIL import Image 

#%% KL divergence related functions
def img2array(imgPath):
    '''Load image as numpy array'''
    image = Image.open(imgPath)
    image = image.convert("L")
    return np.array(image)

def pixelhistogram(gray_array):
    '''Get the pixel histogram for a graycale image'''
    histogram, bin_edges = np.histogram(gray_array, bins=256, range=(0, 256))
    return histogram, bin_edges

def relativeerror(img, ref_img):
    '''Calculate relative error'''
    N = img.shape[0] * img.shape[1]
    error = np.sum(np.abs(img - ref_img))/(np.max(ref_img) - np.min(ref_img))/ N
    return error

def entropy(p_x):
    '''Calculate the entropy of a distribution p(x)'''
    e_sum = 0
    for p in p_x: 
        e = p * np.log2(p) if p != 0 else 0
        e_sum += e
    return -e_sum

def crossentropy(p_x, q_x):
    '''Calculate the cross entropy between distributions p(x), q(x)'''
    e_sum = 0
    for p, q in zip(p_x, q_x):
        if p == 0:
            e = 0
        elif p != 0:
            if q == 0:
                q = 1E-12
            e = p*np.log2(q)
        e_sum += e
    return -e_sum

def relativeentropy(p_x, q_x):
    '''Calculate the relative entropy (or KL divergence) between between distributions p(x), q(x)'''
    return crossentropy(p_x, q_x) - entropy(p_x)
# %%
expPath = 'cexp_data'
recPath = 'rec_data'
samples = ['Ying_Jiang_7', 'Chen_CO', 'Ying_Jiang_1', 'Ying_Jiang_2_1', 
           'Ying_Jiang_2_2', 'Ying_Jiang_3', 'Ying_Jiang_5', 
           'Ying_Jiang_6','Ying_Jiang_4']
# %%
debug = False
mean_KL_values = []
for sample in samples:
    expSamplePath = os.path.join(expPath, sample+'_cro.npz')
    expSample = np.load(expSamplePath)
    print('Experimental Sample: {}, Shape: {}'.format(expSamplePath, expSample['data'].shape))

    recSamplePath = os.path.join(recPath, sample+'_rec.npz')
    recExpSample = np.load(recSamplePath)
    print('Recovered Experimental Sample: {}, Shape: {}'.format(recSamplePath, recExpSample['data'].shape))

    numSlice = expSample['data'].shape[-1]
    if debug:
        fig = plt.subplots(1, numSlice, figsize=(20, 10))
        for i in range(numSlice):
            plt.subplot(1, numSlice, i+1)
            plt.imshow(expSample['data'][:,:,i].T, origin='lower', cmap='gray')
            plt.axis('off')
        plt.show()

    numSlice = recExpSample['data'].shape[-1]
    if debug:
        fig = plt.subplots(1, numSlice, figsize=(20, 10))
        for i in range(numSlice):
            plt.subplot(1, numSlice, i+1)
            plt.imshow(recExpSample['data'][:,:,i].T, origin='lower', cmap='gray')
            plt.axis('off')
        plt.show()

    KL_values = []
    for i in range(numSlice):
        for j in range(numSlice):
            real_afm = expSample['data'][:,:,i]
            # Covert the pixel value to the range of [0, 255]
            real_afm = (real_afm - np.min(real_afm)) / (np.max(real_afm) - np.min(real_afm)) * 255


            histogram_0, bin_edge_0 = pixelhistogram(real_afm)
            p_x = histogram_0 / (real_afm.shape[0] * real_afm.shape[1])  

            rec_afm = recExpSample['data'][:,:,j]
            # Covert the pixel value to the range of [0, 255]
            rec_afm = (rec_afm - np.min(rec_afm)) / (np.max(rec_afm) - np.min(rec_afm)) * 255
            histogram_1, bin_edge_1 = pixelhistogram(rec_afm)
            q_x = histogram_1 / (rec_afm.shape[0] * rec_afm.shape[1])
            if debug and i == 0:
                fig, axs = plt.subplots(1, 2, figsize=(5, 2.5))
                axs[0].bar(bin_edge_0[:-1], p_x, width=bin_edge_0[1]-bin_edge_0[0])
                axs[1].bar(bin_edge_1[:-1], q_x, width=bin_edge_1[1]-bin_edge_1[0])
                plt.show()
            kl = relativeentropy(p_x, q_x)
            KL_values.append(kl)
            if debug:
                print('KL divergence between slice {} and {} is {}'.format(i, j, kl))
    mean_KL = np.mean(KL_values)
    mean_KL_values.append(mean_KL)
    print('Mean KL divergence {} is {}'.format(sample, mean_KL))

# Save the mean KL values in npy and txt format 
mean_KL_values = np.array(mean_KL_values)
print(mean_KL_values)
np.save('mean_KL_values.npy', mean_KL_values)
np.savetxt('mean_KL_values.txt', mean_KL_values)

# %%
