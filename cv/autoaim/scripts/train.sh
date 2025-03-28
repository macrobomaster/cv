#!/usr/bin/env bash

export BASE_PATH=/mnt/tmp/ai/cv-e2e-playground/

export BEAM=3
export IGNORE_JIT_FIRST_BEAM=1
export BEAM_MIN_PROGRESS=5
export BEAM_UOPS_MAX=3000

python -m cv.autoaim.train
