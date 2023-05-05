#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line

docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd), target=/root/home/mambauser/rl-baselines3-zoo,type=bind matinmoezzi/mycobotgym-cpu:latest\
  bash -c "cd /root/home/mambauser/rl-baselines3-zoo && $cmd_line"