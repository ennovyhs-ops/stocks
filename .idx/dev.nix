# To learn more about how to use Nix to configure your environment
# see: https://firebase.google.com/docs/studio/customize-workspace
{ pkgs, ... }: {
  # Which nixpkgs channel to use.
  channel = "stable-24.05"; # or "unstable"

  # Use https://search.nixos.org/packages to find packages
  packages = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.python311Packages.flask
    pkgs.python311Packages.werkzeug
    pkgs.python311Packages.yfinance
    pkgs.python311Packages.pandas
    pkgs.python311Packages.numpy
    pkgs.python311Packages.plotly
    pkgs.python311Packages.requests
    pkgs.python311Packages.polars
    pkgs.python311Packages.numpy-financial
    pkgs.python311Packages.scipy
    pkgs.python311Packages.aiohttp
  ];

  # Sets environment variables in the workspace
}