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
        overlays = common_overlays ++ [
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
            pythonCapWrapper = pkgs-x86_64-linux.stdenv.mkDerivation {
              name = "python-cap-wrapper";
              dontUnpack = true;
              buildInputs = [ pkgs-x86_64-linux.libcap ];
              buildPhase = ''
                cat > wrapper.c << 'EOF'
                #include <sys/prctl.h>
                #include <sys/capability.h>
                #include <unistd.h>
                #include <stdio.h>
                int main(int argc, char *argv[]) {
                  cap_value_t cap_list[] = {CAP_DAC_OVERRIDE, CAP_SYS_RAWIO, CAP_SYS_ADMIN, CAP_IPC_LOCK};
                  int n = sizeof(cap_list) / sizeof(cap_list[0]);
                  cap_t caps = cap_get_proc();
                  if (!caps) { perror("cap_get_proc"); _exit(1); }
                  if (cap_set_flag(caps, CAP_INHERITABLE, n, cap_list, CAP_SET) < 0) { perror("cap_set_flag"); _exit(1); }
                  if (cap_set_proc(caps) < 0) { perror("cap_set_proc"); _exit(1); }
                  cap_free(caps);
                  for (int i = 0; i < n; i++)
                    if (prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_RAISE, cap_list[i], 0, 0) < 0) { perror("prctl"); _exit(1); }
                  execv("${python}/bin/python3", argv);
                  perror("execv");
                  return 1;
                }
                EOF
                $CC wrapper.c -o python-cap-wrapper -lcap
              '';
              installPhase = ''
                mkdir -p $out/bin
                install -m 755 python-cap-wrapper $out/bin/python3
                ln $out/bin/python3 $out/bin/python
              '';
            };
          in
          pkgs-x86_64-linux.mkShell {
            packages =
              with pkgs-x86_64-linux;
              [
                rerun
                pythonEnv
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

              # Set up python environment from withPackages
              export NIX_PYTHONPREFIX='${pythonEnv}'
              export NIX_PYTHONEXECUTABLE='${pythonEnv}/bin/python3'
              export NIX_PYTHONPATH='${pythonEnv}/${python.sitePackages}'

              # Copy the capability wrapper and setcap it
              _CAPS_DIR="$HOME/.cache/python-caps-$(echo '${pythonCapWrapper}' | sha256sum | cut -c1-16)"
              if [ ! -f "$_CAPS_DIR/.ok" ]; then
                rm -rf "$_CAPS_DIR"
                mkdir -p "$_CAPS_DIR"
                cp ${pythonCapWrapper}/bin/python3 "$_CAPS_DIR/python3"
                ln -f "$_CAPS_DIR/python3" "$_CAPS_DIR/python"
                sudo ${pkgs-x86_64-linux.libcap}/bin/setcap 'cap_dac_override,cap_sys_rawio,cap_sys_admin,cap_ipc_lock=ep' "$_CAPS_DIR/python3" && touch "$_CAPS_DIR/.ok"
              fi
              export PATH="$_CAPS_DIR:$PATH"
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
              aravis
              aravis.lib
              gobject-introspection
              llvmPackages_latest.clang-unwrapped
              tmux
              bash
            ];

          shellHook = ''
            export CC=${pkgs-aarch64-linux.llvmPackages_latest.clang-unwrapped}/bin/clang
          '';
        };
      };

      nixosConfigurations = {
        orin-nano-installer = lib.nixosSystem {
          modules = [
            {
              imports = [
                "${inputs.nixpkgs-jetson}/nixos/modules/installer/cd-dvd/installation-cd-minimal.nix"
                inputs.jetpack-nixos.nixosModules.default
              ];
              nixpkgs = {
                buildPlatform = "x86_64-linux";
                hostPlatform = "aarch64-linux";
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
                super = true;
                carrierBoard = "devkit";
                flashScriptOverrides.additionalDtbOverlays = [
                  "${./nix/nixos/enable-serial.dtb}"
                ];
              };
              hardware.deviceTree = {
                enable = true;
                overlays = [
                  {
                    name = "enable-serial";
                    dtsText = ''
                      /dts-v1/;
                      /plugin/;
                      / {
                        fragment@0 {
                          target = <&uarta>;
                          __overlay__ {
                            status = "okay";
                            // serial-tegra RX-DMA UAF (tegra_uart_rx_buffer_push) corrupts
                            // small frames + panics; no "rx"/"tx" in dma-names => force PIO
                            dma-names = "none";
                          };
                        };
                      };
                    '';
                  }
                ];
              };
            }
          ];
        };
        orin-nano = lib.nixosSystem {
          modules = [
            {
              _module.args = { inherit inputs; };
              nixpkgs = {
                buildPlatform = "aarch64-linux";
                hostPlatform = "aarch64-linux";
                config = pkgs-aarch64-linux.config;
              };
              nixpkgs.overlays = common_overlays ++ [
                (final: _: { inherit (final.nvidia-jetpack) cudaPackages; })
              ];

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
                firmware.autoUpdate = true;
                modesetting.enable = true;
                som = "orin-nano";
                super = true;
                carrierBoard = "devkit";
              };

              hardware.deviceTree = {
                enable = true;
                overlays = [
                  {
                    name = "enable-serial";
                    dtsText = ''
                      /dts-v1/;
                      /plugin/;
                      / {
                        fragment@0 {
                          target = <&uarta>;
                          __overlay__ {
                            status = "okay";
                            // serial-tegra RX-DMA UAF (tegra_uart_rx_buffer_push) corrupts
                            // small frames + panics; no "rx"/"tx" in dma-names => force PIO
                            dma-names = "none";
                          };
                        };
                      };
                    '';
                  }
                ];
              };

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

              networking.hostName = "orin-nano";
              system.stateVersion = "25.05";
            }
          ];
        };
      };
    };
}
