import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw


def show(matrix, name):
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(name)
    plt.show()


def create_video():
    if not os.path.exists('frames'):
        os.makedirs('frames')

    img = Image.new('RGB', (500, 500), color='white')
    d = ImageDraw.Draw(img)
    d.ellipse((200, 200, 300, 300), fill='red')
    img.save('frames/000.png')

    for i in range(1, 30):
        img = Image.new('RGB', (500, 500), color='white')
        d = ImageDraw.Draw(img)
        d.ellipse((200, 200 + i*2, 300, 300 + i*2), fill='red')
        img.save('frames/{:03d}.png'.format(i))
