#!/usr/bin/env bash

pushd /root/cv || exit
nix develop . --command env SYSTEM_PATH=/run/cv/sys JITBEAM=4 JIT_BATCH_SIZE=0 FUSE=1 HALF=1 NAV_MAP=./weights/nav_map.json python -m cv.system
