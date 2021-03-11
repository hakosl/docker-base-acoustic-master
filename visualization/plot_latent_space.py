from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imageio
import seaborn
def plot_latent_space(latent_mu, logvar, labels_train, path):
    np_latent_mu = latent_mu.cpu().detach().numpy()
    np_logvar = logvar.cpu().detach().numpy()
    labels_train = labels_train.cpu().detach().numpy()

    fig, axs = plt.subplots(np_logvar.shape[1] + 1, 2, figsize=(5, 2 * (np_logvar.shape[1] + 1)))
    fig.suptitle(path)
    axs[0, 0].set_title("mu density")
    axs[0, 1].set_title("logvar density")
    colors = np.array(["r", "g", "b", "tab:orange", "purple", "cyan", "y"]) 
    axs[np_logvar.shape[1], 0].scatter(np_latent_mu[:, 0], np_latent_mu[:, 1], c=colors[labels_train])
    axs[np_logvar.shape[1], 1].scatter(np_logvar[:, 0], np_logvar[:, 1], c=colors[labels_train])
    for i in range(np_latent_mu.shape[1]):
        
        mi, ma, std = np_latent_mu[:, i].min(), np_latent_mu[:, i].max(), np_latent_mu[:, i].std()
        h = 1.06 * std / ((np_latent_mu.shape[0] ** (1 / 5)))          
        x = np.linspace(mi, ma, 200)
        if h < 1e-10:
            h = 1e-10
        kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(np_latent_mu[:, i].reshape(-1, 1))
        
        x = np.linspace(mi, ma, 200)
        y = kde.score_samples((x.reshape(-1, 1)))
        xmi, xma = y.min(), y.max()
        #print(xmi, xma)
        axs[i, 0].plot(x, np.exp(y), label=f"p(z)", c=colors[0])

        mi, ma, std = np_logvar[:, i].min(), np_logvar[:, i].max(), np_logvar[:, i].std()
        h = 1.06 * std / ((np_latent_mu.shape[0] ** (1 / 5)))   
        x = np.linspace(mi, ma, 200)
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(np_logvar[:, i].reshape(-1, 1))
        

        y = kde.score_samples((x.reshape(-1, 1)))
        axs[i, 1].plot(x, np.exp(y), label=f"p(z)", c=colors[0])
        

    for j in range(labels_train.max()):
        if (~(labels_train == j)).all():
            print(f"no samples {j}")
            continue
        for i in range(np_latent_mu.shape[1]):
            

            
            mi, ma, std = np_latent_mu[:, i].min(), np_latent_mu[:, i].max(), np_latent_mu[:, i].std()
            h = 1.06 * std / ((np_latent_mu.shape[0] ** (1 / 5)))
            if h < 1e-10:
                h = 1e-10
            x = np.linspace(mi, ma, 200)

            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(np_latent_mu[labels_train==j, i].reshape(-1, 1))
            y = kde.score_samples((x.reshape(-1, 1)))

            axs[i, 0].plot(x, np.exp(y), label=f"p(z|x={j})", c=colors[j + 1])

            
            mi, ma, std = np_logvar[:, i].min(), np_logvar[:, i].max(), np_logvar[:, i].std()
            h = 1.06 * std / ((np_latent_mu.shape[0] ** (1 / 5)))       
            x = np.linspace(mi, ma, 200)
            kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(np_logvar[labels_train==j, i].reshape(-1, 1))
            

            y = kde.score_samples((x.reshape(-1, 1)))   
            axs[i, 1].plot(x, np.exp(y), label=f"p(z|x={j})", c=colors[j + 1])

    
    axs[0, 0].legend(["p(z)", "p(z|x=0)", "p(z|x=1)", "p(z|x=2)", "p(z|x=3)", "p(z|x=4)", "p(z|x=5)"], loc ="upper left", bbox_to_anchor=(0.0, 2.4))

    plt.ylabel("Density")
    plt.xlabel("MU / logvar")
    fig.savefig(path)
    plt.close(fig)

def plot_latent_space_gif(filenames, path):
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(path, images)


    