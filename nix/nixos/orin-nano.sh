git clone https://github.com/macrobomaster/cv
nix --experimental-features "nix-command flakes" run github:nix-community/disko/latest -- --mode destroy,format,mount nix/nixos/disk.nix --yes-wipe-all-disks
nixos-install --flake .#orin-nano --root /mnt
