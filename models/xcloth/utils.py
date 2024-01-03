from matplotlib import pyplot as plt
import numpy as np
import copy


def plot_peelmaps(depth_img, rgba_img):
    """
    display peelmaps as image
    """
    fig, ax = plt.subplots(4, 2, figsize=(10, 10))

    for i in range(len(depth_img)):
        # for display, rescale the depth between 0.3 - 0.8
        tmp = depth_img[i][depth_img[i] != 0]
        tmp = np.interp(tmp, (tmp.min(), tmp.max()), (0.3, 0.8))
        d_img = copy.deepcopy(depth_img[i])
        d_img[depth_img[i] != 0] = tmp
        ax[i//2, i%2].imshow(d_img[::-1, ::-1])

        # rgba
        ax[2 + i//2, i%2].imshow(rgba_img[i][::-1, ::-1])

    plt.show()