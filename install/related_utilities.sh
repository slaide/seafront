# install into separate venvs:
# - orange
# - cellprofiler
# - cellprofiler analyst

# requires uv to be already installed

set -euo pipefail

# good to have
sudo apt install -fy gcc clang build-essential cmake make
# for a user
sudo apt install -fy tree curl wget micro
# for orange
sudo apt install -fy libxcb-xinerama0
# for cellprofiler
sudo apt install -fy python3-dev openjdk-11-jdk-headless default-libmysqlclient-dev libsdl2-dev

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64

uv venv venv_orange --python 3.12
source venv_orange/bin/activate
uv pip install PyQt5 PyQtWebEngine
uv pip install orange3
# run with
# source venv_orange/bin/activate
# python -m Orange.canvas
deactivate

wget https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04/wxpython-4.2.3-cp39-cp39-linux_x86_64.whl
uv venv venv_cellprofiler --python 3.9
source venv_cellprofiler/bin/activate
uv pip install --upgrade setuptools wheel
uv pip install wxpython-4.2.3-cp39-cp39-linux_x86_64.whl
rm wxpython-4.2.3-cp39-cp39-linux_x86_64.whl
uv pip install cellprofiler==4.2.8
# run with
# export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# source venv_cellprofiler/bin/activate
# python3 -m cellprofiler
deactivate

wget https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-22.04/wxPython-4.2.1-cp38-cp38-linux_x86_64.whl
uv venv venv_cellprofileranalyst --python 3.8
source venv_cellprofileranalyst/bin/activate
uv pip install --upgrade setuptools wheel numpy
uv pip install --no-build-isolation pandas seaborn scikit-learn verlib python-javabridge python-bioformats
uv pip install wxPython-4.2.1-cp38-cp38-linux_x86_64.whl
rm wxPython-4.2.1-cp38-cp38-linux_x86_64.whl
wget https://github.com/CellProfiler/CellProfiler-Analyst/archive/refs/tags/3.0.5.tar.gz
mv 3.0.5.tar.gz CellProfiler-Analyst-3.0.5.tar.gz
tar -xzf CellProfiler-Analyst-3.0.5.tar.gz
uv pip install --no-build-isolation ./CellProfiler-Analyst-3.0.5
rm CellProfiler-Analyst-3.0.5.tar.gz
# unsafe, but seems to work
sudo ln -s /usr/lib/x86_64-linux-gnu/libtiff.so.6 /usr/lib/x86_64-linux-gnu/libtiff.so.5
# do NOT delete CellProfiler-Analyst-3.0.5 (the folder)
# run with
# source venv_cellprofileranalyst/bin/activate
# python CellProfiler-Analyst-3.0.5/CellProfiler-Analyst.py
deactivate
