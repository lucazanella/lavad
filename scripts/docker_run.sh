#!/bin/bash
docker run --shm-size 64gb --gpus '"device=0,1"' --rm -it -v "$(pwd)":/usr/src/app -v /path/to/directory/datasets:/usr/src/datasets lavad /bin/bash
