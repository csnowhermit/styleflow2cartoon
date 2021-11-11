import os
import cv2
import numpy as np
from PIL import Image


'''
    图像预处理：
    domainB图像为四通道，要转为三通道
'''

'''
    4通道转3通道
    :param img4 np.ndarray, bgr
'''
def channel4to3(img4path):
    img4 = Image.open(img4path)  # PIL.Image，4通道得这样读
    img4 = np.array(img4)
    if img4.shape[2] == 3:    # 三通道的话不用转
        return img4
    A = img4[:, :, 3]
    rgb = img4[:, :, 0:3]
    return rgb

if __name__ == '__main__':
    pathList = ["D:/Documents/work/AI捏脸/ai-人脸2-带头发大图/ai-人脸2/",
                "D:/Documents/work/AI捏脸/ai-人脸-不带头发/",
                "D:/Documents/work/AI捏脸/ai-人脸-带头发/"]
    savepath = "E:/img/"
    if os.path.exists(savepath) is False:
        os.makedirs(savepath)

    count = 0
    for base_path in pathList:
        for file in os.listdir(base_path):
            imgpath = os.path.join(base_path, file)

            rgb = channel4to3(imgpath)
            cv2.imwrite(os.path.join(savepath, "%d.png" % count), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            count += 1