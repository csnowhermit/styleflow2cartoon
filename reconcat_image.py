import os
import cv2
import numpy as np

'''
    根据A2B_*.png，重新拼图
'''

img = cv2.imread("./experiment/A2B_0000980.png")

step = 256
origin_img = []
gen_img = []

for i in range(5, 10):
    tmp = img[0:256, 256 * i:256 * (i+1), :]
    origin_img.append(tmp)
    # cv2.imshow("tmp", tmp)
    # cv2.waitKey()

    tmp2 = img[256*4:256*5, 256 * i:256 * (i+1), :]
    gen_img.append(tmp2)
    # cv2.imshow("tmp2", tmp2)
    # cv2.waitKey()

mask_path = "./experiment/mask_img"
mask_img = []
for file in os.listdir(mask_path):
    if file.startswith("ema_"):
        tmp = cv2.imread(os.path.join(mask_path, file))
        tmp = cv2.resize(tmp, (256, 256))
        mask_img.append(tmp)

tmp_zongxiang = []
for ori, gen, mask in zip(origin_img, gen_img, mask_img):
    tmp = np.concatenate((ori, gen, mask), 0)
    tmp_zongxiang.append(tmp)


tmp1 = np.concatenate([tmp_zongxiang[0], tmp_zongxiang[1], tmp_zongxiang[2], tmp_zongxiang[3], tmp_zongxiang[4]], axis=1)

cv2.imshow("concat_tmp", tmp1)
cv2.waitKey()

cv2.imwrite("./ema_concat_tmp.png", tmp1)


# A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
#                                                                    cam(tensor2numpy(fake_A2A_heatmap[0]),
#                                                                        self.img_size),
#                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
#                                                                    cam(tensor2numpy(fake_A2B_heatmap[0]),
#                                                                        self.img_size),
#                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
#                                                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]),
#                                                                        self.img_size),
#                                                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)),
#                                              1)