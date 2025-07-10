#!/usr/bin/env bash

export MODEL=autoaim

export BEAM_UOPS_MAX=4000
export BEAM_MIN_PROGRESS=5
export BEAM_UPCAST_MAX=256
export BEAM_LOCAL_MAX=1024
export IGNORE_JIT_FIRST_BEAM=1
export JITBEAM=4

export IGNORE_OOB=1
export FUSE_ARANGE=1
# export WINO=1
# export SINGLE_KERNEL_SOFTMAX=1
export DEFAULT_FLOAT=float

python -m cv.train
