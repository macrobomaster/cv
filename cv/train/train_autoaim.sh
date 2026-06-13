#!/usr/bin/env bash

export MODEL=autoaim

export BEAM_UOPS_MAX=4000
export BEAM_MIN_PROGRESS=5
export BEAM_UPCAST_MAX=256
export BEAM_LOCAL_MAX=1024
export JITBEAM=4

export DEFAULT_FLOAT=float

python3 -m cv.train
