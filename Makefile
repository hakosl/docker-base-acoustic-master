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
RUNCMD=docker run $(RUNTIME) --rm --network=host --user $(USERID):$(GROUPID) $(IPC) $(PORT) $(SSHFSOPTIONS) $(DISKS) $(MOUNT) -it $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) $(COMMAND)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip '$(hostname -I)' --port 8888 --allow-root --no-browser --notebook-dir=/acoustic

run:
	$(RUNCMD) python3 main.py --nConv=5

bash:
	$(RUNCMD) bash

vae:
	$(RUNCMD) python3 train_vae.py --config=config/vae_train.yaml


vae-ftest:
	$(RUNCMD) python3 train_vae.py --config=config/vae_test_config.yaml
