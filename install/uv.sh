# install uv (python manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
# create environment (run at proj root)
# see https://docs.astral.sh/uv/pip/environments/
bash -c "uv venv --python 3.13 && uv pip install -e ."