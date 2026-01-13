#!/bin/bash
set -euo pipefail # Don't hide failures
echo "Looking for uv"
if ! [ -x "$(command -v uv)" ]; then
    echo 'uv is not installed. Installing'
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "Found uv installation"
fi

echo "Installing all dependencies with uv."
uv sync --frozen --no-install-project
echo "You can now use the environment in .venv to run notebooks and scripts"
