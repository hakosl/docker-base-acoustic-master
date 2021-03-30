files = ["train_vae.py", "train_vae.py", "supervised_aae.py", "unsupervised_aae.py"]
variables = [{"config": "./config/info_train.yaml" }, {"config": "config/vae_train.yaml"}]

for f, v in zip(files, variables):
    print(f, )
    print(exec(open(f).read(), v))