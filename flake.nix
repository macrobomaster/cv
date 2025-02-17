{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    tinygrad_src = {
      url = "github:tinygrad/tinygrad/76671381aa2fd1f287cbfc5168b5680b64fbf781";
      flake = false;
    };
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
    tinygrad.inputs.tinygrad.follows = "tinygrad_src";

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
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        }
        // nixpkgs_config
      );
    in
    {
      devShells = {
        x86_64-linux.default = pkgs-x86_64-linux.mkShell {
          packages =
            let
              python-packages =
                p: with p; [
                  albumentations
                  opencv4
                  pillow
                  pyvips
                  (tinygrad.override { rocmSupport = true; })
                  wandb
                  pygobject3
                  pygobject-stubs
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
                    nativeBuildInputs = with p; [poetry-core];
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
                ];
              python = pkgs-x86_64-linux.python312;
            in
            with pkgs-x86_64-linux;
            [
              (python.withPackages python-packages)
              aravis
              aravis.lib
              gobject-introspection
              llvmPackages_latest.clang-unwrapped
            ];
        };
        aarch64-linux.default = pkgs-aarch64-linux.mkShell {
          packages =
            let
              python-packages =
                p: with p; [
                  opencv4
                  pillow
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
                  pygobject3
                  pygobject-stubs
                  onnx
                ];
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
            ];
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
