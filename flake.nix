{
  description = "mpi_rust derivation (converted from default.nix)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
    in
    {
      packages.${system}.default = pkgs.stdenv.mkDerivation {
        name = "mpi_rust";

        buildInputs = with pkgs; [
          clang
          llvmPackages.libclang.lib
          cfitsio
          pkg-config


          autoconf
          automake
          libtool
          cmake
          xorg.libX11
          xorg.libXrandr
          xorg.libXinerama
          xorg.libXcursor
          xorg.libXxf86vm
          xorg.libXi
          libGL
          libGL.out
          libGLU
          libGLU.out
          freeglut
          freeglut.out
          libsForQt5.qt5.qtwayland
        ];

        hardeningDisable = [ "all" ];

        # 环境变量也放在 derivation 里即可
        LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";
        LD_LIBRARY_PATH = "${pkgs.libGL}/lib";
        QT_QPA_PLATFORM = "wayland";

        # 如果没有源码，就放置一个空目录
        src = ./.;
      };

      # `nix build` 默认构建上述 derivation
      defaultPackage.${system} = self.packages.${system}.default;
    };
}
