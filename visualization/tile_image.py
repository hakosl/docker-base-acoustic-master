import numpy as np
def tile_sampling(output, n = 6):
    simg = output.cpu().numpy()
    simg = simg[:n]
    n_img, channels, width, height = simg.shape
    simg = simg.reshape((n_img, channels*height, width))
    simg = simg.transpose((0, 2, 1))
    simg = simg.reshape((n_img * width, channels * height))
    simg = simg.transpose((1, 0))
    return simg

def tile_recon(inputs, outputs, labels, n = 3):
    lab = labels.cpu().numpy()[np.newaxis]
    lab[lab < 0] = -1
    lmax, lmin = lab.max(), lab.min()
    lab = (lab - lmin) / (lmax - lmin)

    simg = inputs.cpu().numpy()
    rimg = outputs.data.cpu().numpy()

    simg = simg.transpose((1, 0, 2, 3))
    simg = np.append(simg, lab, axis=0)
    simg = simg.transpose((1, 0, 2, 3))

    rimg = rimg.transpose((1, 0, 2, 3))
    rimg = np.append(rimg, lab, axis=0)
    rimg = rimg.transpose((1, 0, 2, 3))

    ss = simg.shape
    nimg = np.empty((ss[0] * 2, ss[1], ss[2], ss[3]))
    nimg[::2] = simg
    nimg[1::2] = rimg

    nimg = nimg[:n * 2]
    n_img, channels, width, height = nimg.shape
    nimg = nimg.reshape((n_img, channels*height, width))
    nimg = nimg.transpose((0, 2, 1))
    nimg = nimg.reshape((n_img * width, channels * height))
    nimg = nimg.transpose((1, 0))
    return nimg
    fig, axs = plt.subplots(1, 1)
    axs.imshow(nimg, aspect="auto")

    fig.show()