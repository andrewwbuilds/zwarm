#!/bin/zsh
# Installs nerfstudio (for the NeRF map) into ~/.local/share/nerf-env, kept apart from the rest of the project.
# It needs a small shim because nerfstudio 1.1.5 predates a Python 3.11 rule about dataclass defaults.
set -e
ENV=$HOME/.local/share/nerf-env
mkdir -p $ENV && cd $ENV
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python nerfstudio "numpy<2"
cp "${0:A:h}/nerfstudio_sitecustomize.py" "$(.venv/bin/python -c 'import site;print(site.getsitepackages()[0])')/sitecustomize.py"
.venv/bin/ns-train --help > /dev/null 2>&1 && echo "nerfstudio is ready in $ENV"
