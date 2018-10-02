from PIL import Image, ImageDraw
import numpy as np
import os


frames_path = os.path.join(os.path.curdir, 'frames')
if not os.path.exists(frames_path):
    os.makedirs(frames_path)

filename_iter = 0
for i in range(0, 80, 2):
    img = Image.new('RGB', (100, 100), 'white')
    img_draw = ImageDraw.Draw(img)

    # img_draw.ellipse((0+i, 0+i, 20+i, 20+i), fill='green')
    img_draw.ellipse((0+i, 40, 20+i, 60), fill='green')

    img.save(os.path.join(frames_path, '{}.png'.format(filename_iter)))
    filename_iter += 1
