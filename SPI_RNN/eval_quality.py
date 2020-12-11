import numpy as np
import math
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

def psnr(x,y):

    return compare_psnr(x,y)

def ssim(x,y):


    return compare_ssim(x,y,data_range=np.max(y) - np.min(y))


def psnr_average(x, y):

    num = x.shape[0]
    value_list = []

    for i in range(num):
        value = compare_psnr(x[i],y[i])
        value_list.append(value)

    return np.sum(np.array(value_list)) / num

def ssim_average(x,y):

    num = x.shape[0]
    value_list = []

    for i in range(num):
        value = compare_ssim(x[i],y[i],data_range=np.max(y[i]) - np.min(y[i]))
        value_list.append(value)

    return np.sum(np.array(value_list)) / num