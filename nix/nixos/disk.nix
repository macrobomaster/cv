{
  disko.devices.disk.main = {
    device = "/dev/nvme0n1";
    type = "disk";
    content = {
      type = "gpt";
      partitions = {
        ESP = {
          end = "1G";
          type = "EF00";
          content = {
            type = "filesystem";
            format = "vfat";
            mountpoint = "/boot";
            mountOptions = [ "umask=0077" ];
          };
        };
        root = {
          name = "root";
          end = "-16G";
          content = {
            type = "filesystem";
            format = "f2fs";
            mountpoint = "/";
            extraArgs = [
              "-O"
              "extra_attr,inode_checksum,sb_checksum,compression"
            ];
            mountOptions = [
              "compress_algorithm=zstd:6,compress_chksum,atgc,gc_merge,lazytime,nodiscard"
            ];
          };
        };
        swap = {
          size = "100%";
          content = {
            type = "swap";
            discardPolicy = "both";
            resumeDevice = true;
          };
        };
      };
    };
  };
}
