#! /bin/bash

for for c in 2 5 10 20 40 80
do  
   python3 train_aae.py --latent_dim=$c
   python3 aae_semisupervised.py --latent_dim=$c
done


python3 train_vae.py -c=config/vae_train.yaml
python3 train_vae.py -c=config/info_train.yaml

