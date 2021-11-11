import os

# --photo_path ./images/0020.png --save_path ./images/cartoon_0020.png

input_image = ['0016.png',
               '0017.png',
               '0018.png',
               '0019.png',
               '0020.png']

for img in input_image:
    print("python test.py --photo_path ./images/%s --save_path ./images/ema_cartoon_%s" % (img, img))
    os.system("python test.py --photo_path ./images/%s --save_path ./images/ema_cartoon_%s" % (img, img))