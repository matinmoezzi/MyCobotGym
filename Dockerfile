ARG PARENT_IMAGE
FROM $PARENT_IMAGE
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

ENV CODE_DIR /home/$MAMBA_USER

USER root

RUN apt-get -y update && \ 
    apt-get -y install git gcc g++ swig cmake ffmpeg build-essential --reinstall

# Install micromamba env and dependencies
RUN micromamba install -n base -y python==3.10 ale-py swig -c conda-forge && \
    micromamba shell init --shell=bash --prefix=~/micromamba && \
    eval "$(micromamba shell hook --shell=bash)" && \
    micromamba activate base

RUN cd ${CODE_DIR} && \
    git clone https://github.com/matinmoezzi/rl-baselines3-zoo.git && \
    cd rl-baselines3-zoo && \ 
    pip install "autorom[accept-rom-license]" && \
    pip install -r requirements.txt


RUN cd ${CODE_DIR}/rl-baselines3-zoo && \
    pip install -e .

# COPY . ${CODE_DIR}/MyCobotGym

RUN cd ${CODE_DIR} && \
    git clone https://github.com/matinmoezzi/MyCobotGym.git && \
    cd MyCobotGym && \
    pip install -e . && \
    pip cache purge && \
    micromamba clean --all --yes

CMD /bin/bash