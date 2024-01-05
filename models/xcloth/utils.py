from matplotlib import pyplot as plt
import numpy as np
import copy


def plot_depth(depth_img):
    _, ax = plt.subplots(1, 4, figsize=(32, 8))

    for i in range(len(depth_img)):
        # for display, rescale the depth between 0.3 - 0.8
        tmp = depth_img[i][depth_img[i] != 0]
        tmp = np.interp(tmp, (tmp.min(), tmp.max()), (0.3, 0.8))
        d_img = copy.deepcopy(depth_img[i])
        d_img[depth_img[i] != 0] = tmp
        ax[i].imshow(d_img, origin="lower")

    plt.show()


def plot_3d(img):
    _, ax = plt.subplots(1, 4, figsize=(32, 8))

    for i in range(len(img)):
        ax[i].imshow(np.moveaxis(img[i], 0, -1), origin="lower")
    
    plt.show()


def plot_peelmaps(depth_img, rgba_img, norm_img):
    """
    display peelmaps as image
    """
    fig, ax = plt.subplots(6, 2, figsize=(10, 10))

    for i in range(len(depth_img)):
        # for display, rescale the depth between 0.3 - 0.8
        tmp = depth_img[i][depth_img[i] != 0]
        tmp = np.interp(tmp, (tmp.min(), tmp.max()), (0.3, 0.8))
        d_img = copy.deepcopy(depth_img[i])
        d_img[depth_img[i] != 0] = tmp
        ax[i//2, i%2].imshow(d_img, origin="lower")

        # rgb
        ax[2 + i//2, i%2].imshow(np.moveaxis(rgba_img[i], 0, -1), origin="lower")

        # normals
        ax[4 + i//2, i%2].imshow(np.moveaxis(norm_img[i], 0, -1), origin="lower")

    plt.show()