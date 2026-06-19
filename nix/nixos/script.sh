#!/usr/bin/env bash

pushd /root/cv || exit
nix develop . --command env JITBEAM=4 JIT_BATCH_SIZE=0 FUSE=1 HALF=1 python -m cv.system
