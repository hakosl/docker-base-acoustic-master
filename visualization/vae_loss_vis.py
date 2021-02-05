import matplotlib.pyplot as plt
import numpy as np
def vae_loss_visualization(iteration, losses, recon_losses, kl_losses):
    fig, ax = plt.subplots()
    ax.plot(iteration, np.array(losses))
    ax.plot(iteration, np.array(recon_losses))
    ax.plot(iteration, np.array(kl_losses))
    fig.suptitle("VAE loss graph")
    ax.set_xlabel("iteration")
    ax.set_ylabel("loss")
    ax.legend(["recon + kl", "reconstruction", "kl divergence"])
    return fig
