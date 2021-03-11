import numpy as np
import torch
import matplotlib.pyplot as plt
import hdbscan


def make_grid(echogram, model, cm, clf, device, data_transform, window_size = 64, path="output/vae_figures/grid_clustering.png"):


    #data in: H X W X C
    data = echogram.data_numpy(frequencies=[18, 38, 120, 200])
    labels = np.array(echogram.label_memmap())
    labels.astype('float32')
    labels[labels < 0] = -1
    lmin, lmax = labels.min(), labels.max()
    labels = (labels - lmin) / (lmax - lmin)
    labels = labels[:, :, np.newaxis]
    data, _, _, _ = data_transform(data, None, None, None)
    min_depth, max_depth = echogram.range_vector[0], echogram.range_vector[-1]
    y_len = echogram.shape[0]

    # add depthmap
    depthmap = np.repeat(np.linspace(min_depth, max_depth, data.shape[0])[np.newaxis], data.shape[1], axis=0)[np.newaxis].transpose((2, 1, 0))
    depthmap = np.tanh(depthmap/100)
    print(depthmap.shape)
    print(data.shape)
    data = np.concatenate((data, depthmap), axis=2)

    # cut edges so W = window_size * n and H = window_size * m
    data = data[data.shape[0] % window_size:, data.shape[1] % window_size:]
    labels = labels[labels.shape[0] % window_size:, labels.shape[1] % window_size:]
    orig_shape = data.shape

    gridded_data = []
    # 
    for i in range(0, data.shape[0], window_size):
        for j in range(0, data.shape[1], window_size):
            gridded_data.append(data[np.newaxis, i: i + window_size, j:j + window_size])

    # n_img * H * W * C
    # n_img: W/64 * H/64
    orig_data = data.copy()
    data = np.concatenate(gridded_data)

    # new_shape = (-1, window_size, window_size, 5)

    # #data = data.transpose((1, 0, 2))
    # data = data.reshape((window_size, -1, orig_shape[1], 5))
    # data = data.transpose((0, 2, 1, 3))
    # data = data.reshape(new_shape)

    # data = data.reshape((-1, window_size, window_size, 5))
    

    # ax.imshow(data[0, :, :, 0], aspect="auto")

    # fig.suptitle(path)
    # fig.savefig(path)
    # plt.close(fig)
    data = data.transpose((0, 3, 2, 1))

    data = torch.tensor(data).float().to(device)

    # data: n_img * C * W * H
    latent, logvar = model.encoder(data)

    latent = latent.data.cpu().numpy()
    

    clusterings, s = hdbscan.approximate_predict(cm, latent)
    clusterings = clusterings
    
    data = data.data.cpu().numpy()
    
    # color squares according to what cluster they belong to
    c = 0
    for i in range(0, orig_data.shape[0], window_size):
        for j in range(0, orig_data.shape[1], window_size):
            orig_data[i: i + window_size, j: j + window_size] += clusterings[c][np.newaxis, np.newaxis, np.newaxis]       
            c += 1


    #data += clusterings.astype(float)[:, np.newaxis, np.newaxis, np.newaxis]
    data = orig_data
    data = np.concatenate([data, labels], axis=2)

    

    data = data.transpose((2, 0, 1))
    data = data.reshape((-1, data.shape[2]))


    #data = data.reshape((data.shape[0] * data.shape[2], data.shape[1]))
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    ax.imshow(data[:, :], aspect="auto")

    ax.set_yticks(np.arange(0, data.shape[0], window_size))
    ax.set_xticks(np.arange(0, data.shape[1], window_size))

    ax.grid(color='w', linestyle='-', linewidth=0.3)
    fig.suptitle(path)
    fig.savefig(path)
    plt.close(fig)

    return clusterings

