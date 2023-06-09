# 高斯滤波器(Gaussian Filter)  https://blog.csdn.net/ixuyn/article/details/123357809
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import pylab

"""
 高斯滤波器
高斯滤波器/滤波器的作用：
      高斯滤波器将中心像素周围的像素按照高斯分布加权平均进行平滑化。这样的（二维）权值通常被称为卷积核（kernel）或者滤波器（filter）

但是，由于图像的长宽可能不是滤波器大小的整数倍，因此我们需要在图像的边缘补0 。这种方法称作Zero Padding

权值g （卷积核）要进行归一化操作（∑ g = 1 \sum\ g = 1∑ g=1）

高斯掩膜的求解与位置(x,y)无关，因为在计算过程中x,y被抵消掉了,所以求一个kerne模板即可

gaussian_filter算法过程：
1.对图像进行zero padding
2.根据高斯滤波器的核大小和标准差大小实现高斯滤波器
3.使用高斯滤波器对图像进行滤波（相乘再相加）
4。输出高斯滤波后的图像


源码地址
https://github.com/Xnhyacinth/xnhyacinth/blob/master/CV/Gaussian%20Filter.py
"""


# read the image
img = cv2.imread("./image/img1.jpg")

# Print out the type of image data and its dimensions (height, width, and color)
print('This image is:', type(img),
      ' with dimensions:', img.shape)

img_copy = np.copy(img)
#change color to rgb(from bgr)
img_copy = cv2.cvtColor(img_copy,cv2.COLOR_BGR2RGB)


# 添加高斯噪声
def noise_Gaussian(img, mean, var):
      '''
      img:原始图像
      mean：均值
      var：方差，值越大噪声越大
      '''
      # 创建均值为mean方差为var呈高斯分布的图像矩阵
      noise = np.random.normal(mean, var ** 0.5, img.shape)

      # 将原始图像的像素值进行归一化，除以255使像素值在0-1之间
      img = np.array(img / 255, dtype=float)

      # 噪声和图片合并即加噪后的图像
      out = img + noise

      # 解除归一化，乘以255将加噪后的图像的像素值恢复
      out = np.uint8(out * 255)
      return out

# 高斯滤波
def gaussian_filter(img, K_size=3, sigma=1.0):
      img = np.asarray(np.uint8(img))
      if len(img.shape) == 3:
            H, W, C = img.shape
      else:
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

      ## Zero padding
      pad = K_size // 2
      out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float).astype(float)
      out[pad: pad + H, pad: pad + W] = img.copy().astype(float).astype(float)

      print("start")

      ## prepare Kernel
      K = np.zeros((K_size, K_size), dtype=float)
      for x in range(-pad, -pad + K_size):
            for y in range(-pad, -pad + K_size):
                  K[y + pad, x + pad] = np.exp(-(x ** 2 + y ** 2) / (2 * (sigma ** 2))).astype(float)
      K /= (2 * np.pi * sigma * sigma)
      K /= K.sum()
      tmp = out.copy()

      print("zhong")

      # filtering
      for y in range(H):
            for x in range(W):
                  for c in range(C):
                        out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
      out = np.clip(out, 0, 255)
      out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
      print("end")
      return out


# 基于PLT的图像合并，用于比较
def PLT_stack(jpg1, jpg2):
    img1 = Image.open(jpg1)
    img2 = Image.open(jpg2)

    result = Image.new(img1.mode, (640 * 2, 480))
    result.paste(img1, box=(0, 0))
    result.paste(img2, box=(640, 0))
    # result.save("./image/new_image.jpg")
    plt.imshow(result)
    plt.show()

# 基于opencv的图像合成
def opencv_stack(ipg1, jpg2):
    # 原图
    img1 = cv2.imread(jpg1)
    img2 = cv2.imread(jpg2)
    # 灰色图
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ====使用numpy的数组矩阵合并concatenate======
    # 纵向连接
    # image = np.vstack((gray1, gray2))
    # 横向连接
    image = np.concatenate([img1, img2], axis=1)
 # =============
    cv2.imwrite('new_image.jpg', image)
    cv2.imshow('image', image)
    cv2.waitKey(0)


"""
整体高斯滤波展示流程：
1.导入图片（31）
2.添加噪点，用于更加明显的显示（44）
      但是对于image中的img1和img2图片来说，就是滤波之后展示的对比图片
3.进行高斯滤波（64）
4.转换后RBG颜色的原图为img1.jpg，以及滤波后的图片为img2.jpg（147）
5.拼接原图和滤波图（102或114）
"""
# 在这里可以选择下方的不同参数，选择平滑的参数值
out_20_5=gaussian_filter(img_copy,K_size=5,sigma=2.0)
# out_05 = gaussian_filter(img_copy,sigma=0.5)

#  <class 'numpy.ndarray'> -> cv2.jpg 转换数据格式，方便后续基于PLT的合并
cv2.imwrite('img1.jpg', img_copy)
cv2.imwrite('img2.jpg', out_20_5)

img1 = "./image/img1.jpg"
img2 = "./image/img2.jpg"
PLT_stack(img1, img2)


# 高斯过滤参数选择
#display原图
# plt.imshow(img_copy)
# pylab.show()

# # 添加高斯噪声
# out = noise_Gaussian(img_copy, mean=0, var=0.003)
# plt.imshow(out)
# pylab.show()

#sigma=0.5 k_size=3
# out_05=gaussian_filter(img_copy,sigma=0.5)
# plt.imshow(out_05)
# pylab.show()



# #sigma=0.5 k_size=5
# out_05_5=gaussian_filter(img_copy,K_size=5,sigma=0.5)
# plt.imshow(out_05_5)
# pylab.show()
#
# #sigma=0.8 k_size=3
# out_08=gaussian_filter(img_copy,sigma=0.8)
# plt.imshow(out_08)
# pylab.show()
#
# #sigma=0.8 k_size=5
# out_08_5=gaussian_filter(img_copy,K_size=5,sigma=0.8)
# plt.imshow(out_08_5)
# pylab.show()
#
# #sigma=1.3  k_size=3
# out_13=gaussian_filter(img_copy,sigma=1.3)
# plt.imshow(out_13)
# pylab.show()
#
# #sigma=1.3  k_size=5
# out_13_5=gaussian_filter(img_copy,K_size=5,sigma=1.3)
# plt.imshow(out_13_5)
# pylab.show()
#
# #sigma=2.0  k_size=3
# out_20=gaussian_filter(img_copy,K_size=3,sigma=2.0)
# plt.imshow(out_20)
# pylab.show()
#
# #sigma=2.0  k_size=5
# out_20_5=gaussian_filter(img_copy,K_size=5,sigma=2.0)
# plt.imshow(out_20_5)
# pylab.show()






