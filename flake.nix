{
  description = "";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    tinygrad.url = "github:wozeparrot/tinygrad-nix";
  };

  outputs =
    inputs@{
      nixpkgs,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
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
          ];
        };
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
                ];
              python = pkgs.python312;
            in
            with pkgs;
            [
              (python.withPackages python-packages)
              aravis
              aravis.lib
              gobject-introspection
            ];
        };
      }
    );
}
