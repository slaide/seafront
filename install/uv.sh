# install uv (python manager) if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
else
    echo "uv is already installed, skipping installation"
fi
# create environment (run at proj root)
# see https://docs.astral.sh/uv/pip/environments/
bash -c "uv venv --python 3.13 && uv pip install -e ."
