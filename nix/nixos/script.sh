#!/usr/bin/env bash

pushd /root/cv || exit
nix develop . --command env JITBEAM=2 python -m cv.system
