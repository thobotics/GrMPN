FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# Install pip
RUN apt-get update
RUN apt-get -y install python3 python3-pip python3-dev python3-tk
RUN apt-get -y install libglu1-mesa libxi-dev libxmu-dev libglu1-mesa-dev

# Install basic libraries
RUN pip3 install --upgrade pip
RUN pip3 install numpy matplotlib scipy scikit-learn future
RUN pip3 install tensorflow-gpu==1.12 tensorflow-probability==0.5.0 dm-sonnet==1.32 graph_nets jupyter
RUN pip3 install pygsp pyyaml shapely rasterio

# Install additional requirements
RUN pip3 install datetime gitpython h5py tqdm dotmap cython

# Environment setup
RUN echo 'LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu' >> /root/.bashrc
RUN echo 'alias python=python3' >> /root/.bashrc

CMD /bin/bash
