{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
    jetpack-nixos.url = "github:tiiuae/jetpack-nixos/final-stretch";
    disko.url = "github:nix-community/disko/latest";
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-utils,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      nixpkgs_config = {
        overlays = [
          inputs.tinygrad.overlays.default
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                opencv4 = python-prev.opencv4.override {
                  enableGtk3 = true;
                };
              })
            ];
          })
          (final: prev: { makeModulesClosure = x: prev.makeModulesClosure (x // { allowMissing = true; }); })
        ];
      };

      pkgs-x86_64-linux = import nixpkgs (
        {
          system = "x86_64-linux";
        }
        // nixpkgs_config
      );
      pkgs-aarch64-linux = import nixpkgs (
        {
          system = "aarch64-linux";
        }
        // nixpkgs_config
      );
    in
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs (
          {
            inherit system;
          }
          // nixpkgs_config
        );
      in
      {
        devShell = pkgs.mkShell {
          packages =
            let
              python-packages =
                p: with p; [
                  albumentations
                  opencv4
                  pillow
                  (tinygrad.override { rocmSupport = true; })
                  wandb
                  pygobject3
                  pygobject-stubs
                  onnx
                  onnxruntime
                  torchvision
                ];
              python = pkgs.python312;
            in
            with pkgs;
            [
              (python.withPackages python-packages)
              aravis
              aravis.lib
              gobject-introspection
              llvmPackages_latest.clang
            ];
        };
      }
    )
    // {
      nixosConfigurations = {
        orin-nano-installer = pkgs-x86_64-linux.pkgsCross.aarch64-multiplatform.nixos {
          imports = [
            "${nixpkgs}/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix"
            inputs.jetpack-nixos.nixosModules.default
          ];
          hardware.enableAllHardware = lib.mkForce false;
          hardware.nvidia-jetpack = {
            enable = true;
            som = "orin-nano";
            carrierBoard = "devkit";
          };
        };
        orin-nano = pkgs-aarch64-linux.nixos {
          _module.args = { inherit inputs; };
          nixpkgs = nixpkgs_config;

          imports = [
            inputs.jetpack-nixos.nixosModules.default
            ./nix/nixos/base.nix
          ];

          boot.initrd.supportedFilesystems = [ "f2fs" ];
          boot.initrd.availableKernelModules = [
            "nvme"
            "usb_storage"
          ];
          boot.supportedFilesystems = [ "f2fs" ];
          boot.loader.systemd-boot.enable = true;
          boot.loader.efi.canTouchEfiVariables = true;

          fileSystems."/" = {
            device = "/dev/nvme0n1p2";
            fsType = "f2fs";
            options = [
              "compress_algorithm=zstd:6"
              "compress_chksum"
              "atgc"
              "gc_merge"
              "lazytime"
              "nodiscard"
            ];
          };

          fileSystems."/boot" = {
            device = "/dev/nvme0n1p1";
            fsType = "vfat";
            options = [
              "umask=0077"
            ];
          };

          swapDevices = [
            { device = "/dev/nvme0n1p3"; }
          ];

          hardware.enableAllHardware = lib.mkForce false;
          hardware.nvidia-jetpack = {
            enable = true;
            som = "orin-nano";
            carrierBoard = "devkit";
          };

          system.stateVersion = "25.05";
        };
      };
    };
}
