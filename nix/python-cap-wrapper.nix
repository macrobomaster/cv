{ pkgs, python }:
let
  caps = "cap_dac_override,cap_sys_rawio,cap_sys_admin,cap_ipc_lock=ep";
  wrapper = pkgs.stdenv.mkDerivation {
    name = "python-cap-wrapper";
    dontUnpack = true;
    buildInputs = [ pkgs.libcap ];
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
    passthru.setup = pkgs.writeText "setup-caps.sh" ''
      _CAPS_DIR="$HOME/.cache/python-caps-$(echo '${wrapper}' | sha256sum | cut -c1-16)"
      if [ ! -f "$_CAPS_DIR/.ok" ]; then
        rm -rf "$_CAPS_DIR"
        mkdir -p "$_CAPS_DIR"
        cp ${wrapper}/bin/python3 "$_CAPS_DIR/python3"
        ln -f "$_CAPS_DIR/python3" "$_CAPS_DIR/python"
        sudo ${pkgs.libcap}/bin/setcap '${caps}' "$_CAPS_DIR/python3" && touch "$_CAPS_DIR/.ok"
      fi
      export PATH="$_CAPS_DIR:$PATH"
    '';
  };
in
wrapper
