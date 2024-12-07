# Current issues

There are 3 workflows that were explored:
- TAO on Colab
- TAO Launcher 
- TAO using Containers

## TAO on Colab

Even it install the TensorRT and dependencies it still fails at the commands:

!tao model yolo_v4 dataset_convert


## TAO Launcher

TAO launcher is having issues with not detecting or having conflicts with nvidia-container. For some reason even te nvidia container toolkit is installed the TAO toolkit scripts seems to have issues and review is needed.


## TAO using Containers

TAO has issues using Docker-in-Docker and Docker-outside-Docker. This is making extremely hard to develop TAO with workflows that depend on using Containers like Devcontainers and K8s drive workflows.

