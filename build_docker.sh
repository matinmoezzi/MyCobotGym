#!/bin/bash

CPU_PARENT=mambaorg/micromamba:1.4-kinetic
GPU_PARENT=mambaorg/micromamba:1.4.1-focal-cuda-11.7.1

TAG=matinmoezzi/mycobotgym
VERSION=$(cat ./mycobotgym/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
else
  PARENT=${CPU_PARENT}
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT}  -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi