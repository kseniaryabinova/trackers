import matplotlib.pyplot as plt


def show(matrix, name):
    plt.imshow(matrix)
    plt.colorbar()
    plt.title(name)
    plt.show()
