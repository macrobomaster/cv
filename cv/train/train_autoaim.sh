#!/usr/bin/env bash

export MODEL=autoaim

export BEAM_UOPS_MAX=4000
export BEAM_MIN_PROGRESS=5
export BEAM_UPCAST_MAX=256
export BEAM_LOCAL_MAX=1024
export IGNORE_JIT_FIRST_BEAM=1
export BEAM=3

# export WINO=1
# export SINGLE_KERNEL_SOFTMAX=1

python -m cv.train
