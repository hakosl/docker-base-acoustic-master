# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = hos008-docker-test
CONFIG    = pytorch
COMMAND   = python3 unsupervised_img_seg.py
DISKS     = -v $(shell pwd):/app -v /data/deep/data:/data:ro -v /data/hos008/memmap:/memmap -v /data/hos008/acoustic:/acoustic
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)
IPC 	  = --ipc=host
PORT      = -p 8888:8888
MOUNT 	  = #--mount source=memmap,target=/data/memmap:ro
RUNTIME   = --gpus all
ENV 	  = --env-file ./env.list
# No need to change anything below this line

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

.docker: docker/Dockerfile-$(CONFIG)
	docker build $(USERCONFIG) -t $(USERNAME)-$(IMAGENAME) -f docker/Dockerfile-$(CONFIG) --network=host docker

# Using -it for interactive use
RUNCMD=docker run --interactive $(RUNTIME) --rm --network=host --user  $(USERID):$(GROUPID) $(IPC) $(PORT) $(SSHFSOPTIONS) $(DISKS) $(MOUNT) $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) python3 $(C) $(d)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip '$(hostname -I)' --port 8888 --allow-root --no-browser --notebook-dir=/acoustic

run:
	$(RUNCMD) python3 main.py --nConv=5

bash:
	$(RUNCMD) bash

runpy:
	$(RUNCMD) python3 $(C)

vae:
	$(RUNCMD) python3 train_vae.py 

ivae:
	$(RUNCMD) python3 train_vae.py -c=config/info_train.yaml

aae:
	$(RUNCMD) python3 train_aae.py $(d) 

cs:
	$(RUNCMD) python3 compute_stats.py

gns:
	$(RUNCMD) python3 getnumbershools.py

aaess:
	CUDA_VISIBLE_DEVICES=1 $(RUNCMD) python3 unsupervised_aae.py $(d) 


vae-ftest:
	$(RUNCMD) python3 train_vae.py --config=config/vae_test_config.yaml $(d) 

all:
	$(RUNCMD) sh runall.sh

profile:
	$(RUNCMD) python3 -m cProfile -s time train_vae.py -c=config/vae_test_config.yaml

tb:
	$(RUNCMD) tensorboard --logdir=./runs


sd:
	$(RUNCMD) python3 save_dataset.py