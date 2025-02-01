{
  pkgs,
  config,
  inputs,
  lib,
  ...
}:
{
  nix = {
    package = pkgs.nixVersions.latest;
    settings.system-features = [
      "nixos-test"
      "benchmark"
      "big-parallel"
      "kvm"
    ];

    registry =
      let
        nixRegistry = {
          nixpkgs = {
            flake = inputs.nixpkgs;
          };
        };
      in
      lib.mkDefault nixRegistry;

    nixPath = [ "nixpkgs=${inputs.nixpkgs}" ];
    # settings.extra-sandbox-paths = ["/bin/sh=${pkgs.bash}/bin/sh"];

    gc.automatic = false;

    settings = {
      connect-timeout = 5;
      experimental-features = [
        "nix-command"
        "flakes"
      ];
      log-lines = 25;
      max-free = 1024 * 1024 * 1024;
      min-free = 128 * 1024 * 1024;
      builders-use-substitutes = true;
      auto-optimise-store = true;
    };

    daemonCPUSchedPolicy = "batch";
    daemonIOSchedClass = "idle";
    daemonIOSchedPriority = 7;

    settings = {
      substituters = [
        "https://nix-community.cachix.org"
      ];
      trusted-public-keys = [
        "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
      ];
    };
  };

  environment = {
    systemPackages = with pkgs; [
      # core system utils
      binutils
      coreutils

      # extra utils
      bat
      btop
      curl
      du-dust
      fd
      file
      git
      htop
      jq
      less
      neovim
      nix-output-monitor
      pciutils
      ripgrep
      tree
      usbutils

      # network utils
      dnsutils
      nmap
      whois
    ];

    shellAliases = {
      # grep
      grep = "rg";

      # my public ip
      myip = "dig +short myip.opendns.com @208.67.222.222 2>&1";
    };

    sessionVariables = {
      PAGER = "less";
      LESS = "-iFJMRWX -z-4 -x4";
      LESSOPEN = "|${pkgs.lesspipe}/bin/lesspipe.sh %s";
    };
  };

  # extra programs
  programs.fish = {
    enable = true;
  };
  users.defaultUserShell = pkgs.fish;
  programs.command-not-found.enable = false;

  # disable manually creating users
  users.mutableUsers = false;

  # clean tmp on boot
  boot.tmp.cleanOnBoot = true;

  # networking
  systemd.services.NetworkManager-wait-online.enable = false;
  systemd.network.wait-online.enable = false;
  systemd.services.systemd-resolved.stopIfChanged = false;
  networking.useNetworkd = true;
  systemd.network.enable = true;

  # time
  services.timesyncd.enable = lib.mkDefault true;

  # disabled unneeded stuff
  security = {
    polkit.enable = lib.mkDefault false;
    audit.enable = false;
  };
  services.udisks2.enable = false;
  systemd.tpm2.enable = false;
  xdg = {
    autostart.enable = false;
    icons.enable = false;
    mime.enable = false;
    sounds.enable = false;
  };

  # remove docs
  documentation = {
    enable = false;
    doc.enable = false;
    info.enable = false;
    man.enable = false;
    nixos.enable = false;
  };

  # remove unneeded locales
  i18n.supportedLocales = [ (config.i18n.defaultLocale + "/UTF-8") ];

  # remove fonts
  fonts.fontconfig.enable = lib.mkDefault false;

  # enable ssh
  services.openssh.enable = lib.mkForce true;
  users.users.root.openssh.authorizedKeys.keys = [
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKPL+OWmcGo4IlL+LUz9uEgOH8hk0JIN3DXEV8sdgxPB wozeparrot"
  ];
  users.users.root.hashedPassword = "$y$j9T$zj/Hf6NsqCBGuD.Wt0ux0.$fvdcKDDnICCnYGhRGC.I/I0GW/BQNOezvQzNQ1RGUQ3";

  # systemd tweaks
  systemd.enableEmergencyMode = true;
  systemd.sleep.extraConfig = ''
    AllowSuspend=no
    AllowHibernation=no
  '';

  # use systemd initrd
  boot.initrd.systemd.enable = true;
  boot.initrd.systemd.emergencyAccess = true;

  # firewall
  networking = {
    firewall = {
      enable = true;
      allowPing = true;
      allowedTCPPorts = [ 22 ];
    };
  };

  # use tcp bbr
  boot.kernel.sysctl = {
    "net.core.default_qdisc" = "fq";
    "net.ipv4.tcp_congestion_control" = "bbr";
  };
}
