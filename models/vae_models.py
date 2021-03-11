from models.vae_model import vae_models
from models.mmd_vae import InfoVAE


vae_models = {
    **vae_models,
    "InfoVAE": InfoVAE
}