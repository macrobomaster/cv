# cv

## Running

1. Install nix
`curl -fsSL https://install.determinate.systems/nix | sh -s -- install`

2. enter nix shell
`nix develop .`

3. create system dir
`mkdir -p sys`

4. run on pc
`PC=1 WEBCAM=0 python3 -m cv.system`

5. visualize
in another terminal in the nix shell
`rerun`
in another terminal in the nix shell
`python -m cv.tools.visual 127.0.0.1`
