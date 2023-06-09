import cv2 as cv
#高斯金字塔
def pyramid_demo(image):
    level = 3      #设置金字塔的层数为3
    temp = image.copy()  #拷贝图像
    pyramid_images = []  #建立一个空列表
    for i in range(level):
        dst = cv.pyrDown(temp)   #先对图像进行高斯平滑，然后再进行降采样（将图像尺寸行和列方向缩减一半）
        pyramid_images.append(dst)  #在列表末尾添加新的对象
        cv.imshow("pyramid"+str(i), dst)
        temp = dst.copy()
    return pyramid_images
#拉普拉斯金字塔
def lapalian_demo(image):
    pyramid_images = pyramid_demo(image)    #做拉普拉斯金字塔必须用到高斯金字塔的结果
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            expand = cv.pyrUp(pyramid_images[i], dstsize = image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
        else:
            cv.waitKey(0)
            expand = cv.pyrUp(pyramid_images[i], dstsize = pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lapalian_down_"+str(i), lpls)
            cv.waitKey(0)

src = cv.imread('./image/1.JPG')
cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE) #设置为WINDOW_NORMAL可以任意缩放
cv.imshow('input_image', src)
lapalian_demo(src)
cv.waitKey(0)
cv.destroyAllWindows()