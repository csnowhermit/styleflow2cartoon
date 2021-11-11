# styleflow2cartoon：基于styleflow+photo2cartoon实现2D捏脸

## Requirements

streamlit 1.0.0

## Start

## 1.随机生成头像捏脸

``` shell
streamlit run app.py
```

## 2.上传图片实现捏脸

### 2.1.通过图片生成latent code：

见 stylegan2-ada-pytorch 中 projector.py：

``` Shell
python projector.py --outdir=out --target=./test.jpg --network=./pretrained/ffhq.pkl
```

### 2.2.修改app_local.py的npy文件加载方式：

``` Python
# 第238行：
# 这里w_selected直接用npz文件代替
project_path = "./0000.npz"
w = np.load(project_path)
if project_path.endswith(".npz"):
	w_selected = w['w']    # [1, 18, 512]
else:
	w_selected = w
	
# 第279行
# 这里直接读取图片代替
import cv2
img_source = cv2.imread("0000.jpg")
img_source = cv2.resize(img_source, (1024, 1024))
img_source = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
```

### 2.3.启动程序：

``` Shell
streamlit run app_local.py
```

## 参考论文：







