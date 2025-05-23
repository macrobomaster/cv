{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    tinygrad.url = "github:wozeparrot/tinygrad-nix";

    jetpack-nixos.url = "github:tiiuae/jetpack-nixos/final-stretch";
    # jetpack-nixos.url = "github:anduril/jetpack-nixos";
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

      common_overlays = [
        inputs.tinygrad.overlays.default
        (final: prev: { makeModulesClosure = x: prev.makeModulesClosure (x // { allowMissing = true; }); })
      ];

      pkgs-x86_64-linux = import nixpkgs ({
        system = "x86_64-linux";
        overlays = [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                opencv4 = python-prev.opencv4.override {
                  enableGtk3 = true;
                };
              })
            ];
          })
        ] ++ common_overlays;
      });
      pkgs-aarch64-linux = import nixpkgs ({
        system = "aarch64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
        overlays = [
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                opencv4 = python-prev.opencv4.override {
                  enableCuda = false;
                };
              })
            ];
          })
        ] ++ common_overlays;
      });

      common-python-packages =
        p: with p; [
          opencv4
          numpy
          pygobject3
          pygobject-stubs
          pyserial
          pyzmq
          cbor2
          setproctitle
          xxhash
          scipy
        ];
    in
    {
      devShells = {
        x86_64-linux.default = pkgs-x86_64-linux.mkShell {
          packages =
            let
              python-packages =
                p:
                with p;
                [
                  albumentations
                  pillow
                  pyvips
                  (tinygrad.override { rocmSupport = true; })
                  wandb
                  onnx
                  onnxruntime
                  torchvision
                  transformers
                  (p.buildPythonPackage rec {
                    pname = "moondream";
                    version = "0.0.6";
                    pyproject = true;
                    src = pkgs-x86_64-linux.fetchPypi {
                      inherit pname version;
                      hash = "sha256-uSN2dTCvmWkzDRCsTQeNIFZot4SCVm0Jo4lyKYjqaP4=";
                    };
                    pythonRelaxDeps = [
                      "onnxruntime"
                      "pillow"
                      "tokenizers"
                    ];
                    nativeBuildInputs = with p; [ poetry-core ];
                    propagatedBuildInputs = with p; [
                      numpy
                      onnx
                      onnxruntime
                      pillow
                      tokenizers
                      pyvips
                      einops
                    ];
                    doCheck = false;
                  })
                  rerun-sdk
                  gymnasium
                ]
                ++ common-python-packages p;
              python = pkgs-x86_64-linux.python312;
            in
            with pkgs-x86_64-linux;
            [
              (python.withPackages python-packages)
              aravis
              aravis.lib
              gobject-introspection
              llvmPackages_latest.clang-unwrapped
              waypipe
              sqlite-web
              rerun
              (pkgs.writeShellScriptBin "rerun-web" ''
                #!/usr/bin/env bash
                ${rerun}/bin/rerun --web-viewer
              '')
            ];

          shellHook = ''
            export CC=${pkgs-x86_64-linux.llvmPackages_latest.clang-unwrapped}/bin/clang
          '';
        };
        aarch64-linux.default = pkgs-aarch64-linux.mkShell {
          packages =
            let
              python-packages =
                p:
                with p;
                [
                  (
                    (tinygrad.override {
                      cudaSupport = true;
                      torch = null;
                    }).overrideAttrs
                    (_: {
                      doCheck = false;
                      pytestCheckPhase = ''
                        echo "Skipping tests"
                      '';
                    })
                  )
                ]
                ++ common-python-packages p;
              python = pkgs-aarch64-linux.python312;
            in
            with pkgs-aarch64-linux;
            [
              (python.withPackages python-packages)
              aravis
              aravis.lib
              gobject-introspection
              llvmPackages_latest.clang-unwrapped
              cudaPackages.cuda_nvdisasm
              cudaPackages.cuda_nvcc
              cudaPackages.cuda_cccl
              cudaPackages.cuda_cudart
              cudaPackages.cudatoolkit
              gcc13
              tmux
            ];

          shellHook = ''
            export CC=${pkgs-aarch64-linux.llvmPackages_latest.clang-unwrapped}/bin/clang
          '';
        };
      };

      nixosConfigurations = {
        orin-nano-installer = pkgs-x86_64-linux.pkgsCross.aarch64-multiplatform.nixos {
          imports = [
            "${nixpkgs}/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix"
            inputs.jetpack-nixos.nixosModules.default
          ];
          boot.kernelPatches = [
            {
              name = "config";
              patch = null;
              extraConfig = ''
                ARM64_PMEM y
                PCI_TEGRA y
                PCIE_TEGRA194 y
                PCIE_TEGRA194_HOST y
                BLK_DEV_NVME y
                NVME_CORE y
                FB_SIMPLE y
              '';
            }
          ];
          boot.supportedFilesystems = {
            zfs = lib.mkForce false;
          };
          boot.initrd.supportedFilesystems = {
            zfs = lib.mkForce false;
          };
          hardware.enableAllHardware = lib.mkForce false;
          hardware.nvidia-jetpack = {
            enable = true;
            som = "orin-nano";
            carrierBoard = "devkit";
          };
        };
        orin-nano = pkgs-aarch64-linux.nixos {
          _module.args = { inherit inputs; };
          nixpkgs.overlays = common_overlays;

          imports = [
            inputs.jetpack-nixos.nixosModules.default
            inputs.disko.nixosModules.disko
            ./nix/nixos/base.nix
            ./nix/nixos/disk.nix
          ];

          boot.kernelPatches = [
            {
              name = "config";
              patch = null;
              extraConfig = ''
                ARM64_PMEM y
                PCI_TEGRA y
                PCIE_TEGRA194 y
                PCIE_TEGRA194_HOST y
                BLK_DEV_NVME y
                NVME_CORE y
                FB_SIMPLE y
                IWLWIFI m
              '';
            }
          ];
          boot.initrd.availableKernelModules = [
            "nvme"
            "f2fs"
            "pcie-tegra194"
          ];
          boot.supportedFilesystems = [
            "f2fs"
            "vfat"
          ];
          boot.loader.systemd-boot.enable = true;
          boot.loader.efi.canTouchEfiVariables = true;

          hardware.graphics.enable = true;
          hardware.nvidia-jetpack = {
            enable = true;
            modesetting.enable = true;
            som = "orin-nano";
            carrierBoard = "devkit";
          };

          networking.hostName = "orin-nano";
          system.stateVersion = "25.05";
        };
      };
    };
}
