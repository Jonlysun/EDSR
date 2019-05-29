import numpy as np
from scipy import misc
import tensorflow as tf

def psnr(im1, im2):
    """ im1 and im2 value must be between 0 and 255"""
    im1 = np.float64(im1)
    im2 = np.float64(im2)
    rmse = np.sqrt(np.mean(np.square(im1[:] - im2[:])))
    psnr = 20 * np.log10(255 / rmse)
    return psnr


def img_to_uint8(img):
    #unit8 无符号8位整数（0-255）
    if np.max(img) <= 1.0:   #np.max()取出数组中最大的数
        img *= 255.0
    img = np.clip(img, 0, 255)   #小于0的变成0， 大于255的变成255
    return np.round(img).astype(np.uint8)  #round将其四舍五入，astype修改数组类型为0-255


rgb_to_ycbcr = np.array([[65.481, 128.553, 24.966],
                         [-37.797, -74.203, 112.0],
                         [112.0, -93.786, -18.214]])
# RGB——>YCbCr将RGB图像转换为亮度信号和色差信号
ycbcr_to_rgb = np.linalg.inv(rgb_to_ycbcr)


def rgb2ycbcr(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = np.dot(img, rgb_to_ycbcr.T) / 255.0  #为什么采用这种转换的方式，/255之后矩阵正确，也许是为了逆转换方
    img = img + np.array([16, 128, 128])
    return img


def ycbcr2rgb(img):
    """ img value must be between 0 and 255"""
    img = np.float64(img)
    img = img - np.array([16, 128, 128])
    img = np.dot(img, ycbcr_to_rgb.T) * 255.0
    return img


def tf_resize_image(imgs, scale):
    def resize_image(imgs, scale):
        b = imgs.shape[0]
        c = imgs.shape[-1]
        #print(imgs)
        res = []
        for i in range(b):
            img = imgs[i]          
            #print(np.array(img).shape)            #(24,24,1)
            tar_img = []
            for j in range(c):
                tar_img.append(misc.imresize(img[:, :, j], scale / 1.0, 'bicubic', mode='F'))
            #print(np.array(tar_img).shape)        #(1, 96, 96)
            img = np.stack(tar_img, -1)            # -1表示增加的为最高维，现在为3维
            #print(np.array(img).shape)            #(96, 96, 1)
            res.append(img)  #4维
            print(np.array(res).shape)             #(1, 96, 96, 1)

        return np.stack(res)                       #(1, 96, 96, 1)  axis = 0则不改变形状
    return tf.py_func(lambda x: resize_image(x, scale), [imgs], tf.float32)
