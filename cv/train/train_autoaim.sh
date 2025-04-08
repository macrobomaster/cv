#!/usr/bin/env bash

export MODEL=autoaim

export BEAM_UOPS_MAX=3000
export BEAM_MIN_PROGRESS=5
export IGNORE_JIT_FIRST_BEAM=1
export BEAM=3

export WINO=1

python -m cv.train
