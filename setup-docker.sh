#!/bin/bash

# build the simclr container
docker build -f noisy.Dockerfile -t sally20921/noisy:1.0.0 .

# before push, login to your docker account using docker login
sudo docker push sally20921/swav:2.0.0

# start 
# docker run --rm -ti --runtime=nvidia --name simclr sally20921/simclr bash
#docker run --rm --runtime=nvidia 
#	--name swav \
#	--privileged \
#	-v /dev/bus/usb:/dev/bus/usb \
#	-v /dev/dri:/dev/dri \
#	-v "$(pwd)":/home/ \
#	-v /SSD/ILSVRC2012:/home/data/imagenet \
#	-it sally20921/swav:1.0.0 /bin/zsh

# ssh into running containers
# docker ps
# docker exec -ti container_id bash
# default shm is 64mb
docker run --rm --runtime=nvidia --name BYOL --shm-size=2g --gpus=all -v /SSD/ILSVRC2012:/home/data/imagenet -it sally20921/swav:2.0.0 /bin/zsh
