# Change the configuration here.
# Include your useid/name as part of IMAGENAME to avoid conflicts
IMAGENAME = hos008-docker-test
CONFIG    = basic
COMMAND   = python3 train_vae.py
DISKS     = -v /data/deep/data:/data:ro -v /data/håkon/memmap:/memmap -v /data/håkon/acoustic:/acoustic
USERID    = $(shell id -u)
GROUPID   = $(shell id -g)
USERNAME  = $(shell whoami)
PORT      = -p 8888:8888
MOUNT 	  = #--mount source=memmap,target=/data/memmap:ro
RUNTIME   = --gpus all
# No need to change anything below this line

# Allows you to use sshfs to mount disks
SSHFSOPTIONS = --cap-add SYS_ADMIN --device /dev/fuse

USERCONFIG   = --build-arg user=$(USERNAME) --build-arg uid=$(USERID) --build-arg gid=$(GROUPID)

.docker: docker/Dockerfile-$(CONFIG)
	docker build $(USERCONFIG) -t $(USERNAME)-$(IMAGENAME) -f docker/Dockerfile-$(CONFIG) --network=host docker

# Using -it for interactive use
RUNCMD=docker run $(RUNTIME) --rm --network=host --user $(USERID):$(GROUPID) $(PORT) $(SSHFSOPTIONS) $(DISKS) $(MOUNT) -it $(USERNAME)-$(IMAGENAME)

# Replace 'bash' with the command you want to do
default: .docker
	$(RUNCMD) $(COMMAND)

# requires CONFIG=jupyter
jupyter:
	$(RUNCMD) jupyter notebook --ip '$(hostname -I)' --port 8888

run:
	$(RUNCMD) python3 main.py

bash:
	$(RUNCMD) bash
