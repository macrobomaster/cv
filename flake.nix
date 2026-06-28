{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgs-jetson.url = "github:nixos/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";

    tinygrad.url = "github:wozeparrot/tinygrad-nix";
    tinygrad.inputs.nixpkgs.follows = "nixpkgs";

    jetpack-nixos.url = "github:anduril/jetpack-nixos";
    jetpack-nixos.inputs.nixpkgs.follows = "nixpkgs-jetson";
    disko.url = "github:nix-community/disko/latest";
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-utils,
      ...
    }:
    let
      inherit (inputs.nixpkgs-jetson) lib;

      common_overlays = [
        inputs.tinygrad.overlays.default
        (final: prev: { makeModulesClosure = x: prev.makeModulesClosure (x // { allowMissing = true; }); })
      ];

      # aarch64 (jetson) only — kept off x86_64 so it doesn't churn the x86 devshell
      aarch64_overlays = common_overlays ++ [
        # SDL3's checkPhase runs an SDL_CreateProcess test that asserts wrong in the build sandbox
        (final: prev: { sdl3 = prev.sdl3.overrideAttrs (old: { doCheck = false; }); })
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
      pkgs-aarch64-linux = import inputs.nixpkgs-jetson ({
        system = "aarch64-linux";
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaVersion = "12";
          cudaCapabilities = [
            "8.7"
          ];
        };
        overlays = aarch64_overlays ++ [
          inputs.jetpack-nixos.overlays.default
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                opencv4 = python-prev.opencv4.override {
                  enableCuda = false;
                };
                tinygrad = python-prev.tinygrad.override {
                  cudaPackages = final.nvidia-jetpack.cudaPackages;
                };
              })
            ];
          })
        ];
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

      # shared by both Jetson configs (the installer ISO and the on-disk system)
      jetson-common = {
        imports = [ inputs.jetpack-nixos.nixosModules.default ];
        hardware.nvidia-jetpack = {
          enable = true;
          # som is set per-config (orin-nano vs orin-nx) by mkOrin below
          super = true;
          carrierBoard = "devkit";
          # uarta PIO overlay (serial-tegra RX-DMA UAF workaround); UEFI applies it at boot
          flashScriptOverrides.additionalDtbOverlays = [
            "${./nix/nixos/enable-serial.dtb}"
          ];
        };
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

        # board-level: NVMe-over-Tegra-PCIe + on-board filesystems
        boot.initrd.availableKernelModules = [
          "nvme"
          "pcie-tegra194"
        ];
        # btrfs root is auto-included from disk.nix; just need the vfat ESP here
        boot.supportedFilesystems = {
          vfat = true;
        };
      };

      # the installer ISO (cross-built on x86_64); flashes firmware + provisions NVMe for its SoM
      orinInstaller =
        { pkgs, ... }:
        {
          imports = [ "${inputs.nixpkgs-jetson}/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix" ];
          nixpkgs = {
            buildPlatform = "x86_64-linux";
            hostPlatform = "aarch64-linux";
          };
          boot.supportedFilesystems.zfs = lib.mkForce false;
          boot.initrd.supportedFilesystems.zfs = lib.mkForce false;
          hardware.enableAllHardware = lib.mkForce false;
          # partition + install onto NVMe from the ISO; disko CLI from the same input as the module
          environment.systemPackages = [
            inputs.disko.packages.${pkgs.stdenv.hostPlatform.system}.default
            pkgs.git
          ];
        };

      # the on-disk system (native aarch64); hostname tracks the SoM (orin-nano/orin-nx)
      orinSystem =
        { config, ... }:
        {
          _module.args = { inherit inputs; };
          nixpkgs = {
            buildPlatform = "aarch64-linux";
            hostPlatform = "aarch64-linux";
            config = pkgs-aarch64-linux.config;
          };
          nixpkgs.overlays = aarch64_overlays ++ [
            (final: _: { inherit (final.nvidia-jetpack) cudaPackages; })
          ];

          imports = [
            inputs.disko.nixosModules.disko
            ./nix/nixos/base.nix
            ./nix/nixos/disk.nix
          ];

          boot.loader.systemd-boot.enable = true;
          boot.loader.efi.canTouchEfiVariables = true;

          hardware.graphics.enable = true;
          hardware.nvidia-jetpack = {
            firmware.autoUpdate = true;
            modesetting.enable = true;
          };

          # preload the cv devshell's full build closure so `nix develop` at boot is no-build
          system.extraDependencies = [ orinDevShell.inputDerivation ];

          systemd.services.cv = {
            description = "cv service";
            wantedBy = [ "multi-user.target" ];
            after = [ "network.target" ];
            serviceConfig = {
              Type = "oneshot";
              WorkingDirectory = "/root/cv";
              RemainAfterExit = true;
              ExecStart = "${pkgs-aarch64-linux.tmux}/bin/tmux new-session -d -s cv ${./nix/nixos/script.sh}";
              ExecStop = "${pkgs-aarch64-linux.tmux}/bin/tmux kill-session -t cv";
              TimeoutStopSec = 1;
            };
          };

          networking.hostName = config.hardware.nvidia-jetpack.som;
          system.stateVersion = "25.05";
        };

      # one config per SoM; firmware/boardspec differ, everything else is shared
      mkOrin =
        som: extra:
        lib.nixosSystem {
          modules = [
            jetson-common
            { hardware.nvidia-jetpack.som = som; }
          ]
          ++ extra;
        };

      # bake a target system's full closure + its disko partition script into the
      # installer ISO, so provisioning is a copy (no rebuild/eval/network) on the slow Orin
      embedSystem =
        sys:
        { pkgs, ... }:
        {
          isoImage.storeContents = [ sys.config.system.build.toplevel ];
          environment.systemPackages = [
            (pkgs.writeShellScriptBin "install-orin" ''
              set -euo pipefail
              echo ">>> partitioning + formatting nvme (disko, DESTRUCTIVE) ..."
              ${sys.config.system.build.diskoScript}
              echo ">>> installing pre-built system (copy, no build) ..."
              nixos-install --system ${sys.config.system.build.toplevel} --no-root-passwd
              echo ">>> done — reboot"
            '')
          ];
        };

      orinNano = mkOrin "orin-nano" [ orinSystem ];
      orinNx = mkOrin "orin-nx" [ orinSystem ];

      # the aarch64 cv runtime shell (tinygrad+CUDA, opencv, aravis, …); exposed as the
      # devShell AND preloaded into the system closure via .inputDerivation (see orinSystem)
      orinDevShell = pkgs-aarch64-linux.mkShell {
        packages =
          let
            python-packages =
              p:
              with p;
              [
                (
                  (tinygrad.override {
                    cudaSupport = true;
                  }).overridePythonAttrs
                  (old: {
                    doCheck = false;
                    nativeCheckInputs = [ ];
                    # orin NVRTC (cuda 12, aarch64) rejects the union-based tg_bitcast when a
                    # union member is __half ("disallowed member function"), and has no
                    # __builtin_memcpy; bitcast via a pointer reinterpret instead.
                    postPatch = (old.postPatch or "") + ''
                      substituteInPlace tinygrad/renderer/cstyle.py \
                        --replace-fail \
                          "union U { F f; T t; }; U u; u.f = v; return u.t;" \
                          "return *(T*)(&v);"
                    '';
                  })
                )
              ]
              ++ common-python-packages p;
            python = pkgs-aarch64-linux.python313;
          in
          with pkgs-aarch64-linux;
          [
            (python.withPackages python-packages)
            apriltag                       # AprilTag 3 C lib (ctypes-wrapped by tagd)
            aravis
            aravis.lib
            gobject-introspection
            llvmPackages_latest.clang-unwrapped
            tmux
            bash
          ];
        shellHook = ''
          export CC=${pkgs-aarch64-linux.llvmPackages_latest.clang-unwrapped}/bin/clang
          # tagd ctypes-loads libapriltag directly from this absolute store path
          export APRILTAG_LIB=${pkgs-aarch64-linux.apriltag}/lib/libapriltag.so
        '';
      };
    in
    {
      devShells = {
        x86_64-linux.default =
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
                rerun-sdk
                z3-solver
              ]
              ++ common-python-packages p;
            python = pkgs-x86_64-linux.python314;
            pythonEnv = python.withPackages python-packages;
            pythonCapWrapper = import ./nix/python-cap-wrapper.nix {
              pkgs = pkgs-x86_64-linux;
              inherit python;
            };
          in
          pkgs-x86_64-linux.mkShell {
            packages =
              with pkgs-x86_64-linux;
              [
                rerun
                pythonEnv
                apriltag                       # AprilTag 3 C lib (ctypes-wrapped by tagd)
                aravis
                aravis.lib
                gobject-introspection
                llvmPackages_latest.clang-unwrapped
                waypipe
                sqlite-web
                picocom
                tmux
                (pkgs.writeShellScriptBin "rerun-web" ''
                  #!/usr/bin/env bash
                  ${rerun}/bin/rerun --web-viewer
                '')
              ];

            shellHook = ''
              export CC=${pkgs-x86_64-linux.llvmPackages_latest.clang-unwrapped}/bin/clang
              # tagd ctypes-loads libapriltag directly from this absolute store path
              export APRILTAG_LIB=${pkgs-x86_64-linux.apriltag}/lib/libapriltag.so

              # Set up python environment from withPackages
              export NIX_PYTHONPREFIX='${pythonEnv}'
              export NIX_PYTHONEXECUTABLE='${pythonEnv}/bin/python3'
              export NIX_PYTHONPATH='${pythonEnv}/${python.sitePackages}'

              source ${pythonCapWrapper.setup}
            '';
          };
        aarch64-linux.default = orinDevShell;
      };

      nixosConfigurations = {
        orin-nano = orinNano;
        orin-nx = orinNx;
        orin-nano-installer = mkOrin "orin-nano" [ orinInstaller (embedSystem orinNano) ];
        orin-nx-installer = mkOrin "orin-nx" [ orinInstaller (embedSystem orinNx) ];
      };
    };
}
