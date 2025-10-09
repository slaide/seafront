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

# Generate .desktop files for GUI applications
echo "Generating .desktop files on desktop..."

DESKTOP_DIR="$HOME/Desktop"
CURRENT_DIR="$HOME"
SEAFRONT_DIR="$HOME/Documents/seafront"

# Function to create .desktop file if it doesn't exist
create_desktop_file() {
    local filename="$1"
    local content="$2"
    local filepath="$DESKTOP_DIR/$filename"

    if [ -f "$filepath" ]; then
        echo "WARNING: $filepath already exists, skipping creation to avoid overwriting"
        return 1
    else
        echo "$content" > "$filepath"
        chmod +x "$filepath"
        echo "Created: $filepath"
        return 0
    fi
}

# Seafront .desktop file
SEAFRONT_CONTENT='[Desktop Entry]
Name=Seafront
Comment=Open-source microscope control software for SQUID microscopes
Exec=bash -c '\''cd "'"$SEAFRONT_DIR"'" && uv run python -m seafront --microscope "squid"'\''
Icon=applications-science
Terminal=true
Type=Application
Categories=Science;Education;
Keywords=microscope;imaging;science;SQUID;
StartupNotify=true'

create_desktop_file "seafront.desktop" "$SEAFRONT_CONTENT"

# Orange .desktop file
ORANGE_CONTENT='[Desktop Entry]
Name=Orange Data Mining
Comment=Interactive data analysis and machine learning toolkit
Exec=bash -c '\''cd "'"$CURRENT_DIR"'" && source venv_orange/bin/activate && python -m Orange.canvas'\''
Icon=applications-science
Terminal=false
Type=Application
Categories=Science;Education;Development;
Keywords=data;mining;machine;learning;visualization;
StartupNotify=true'

create_desktop_file "orange.desktop" "$ORANGE_CONTENT"

# CellProfiler .desktop file
CELLPROFILER_CONTENT='[Desktop Entry]
Name=CellProfiler
Comment=Open-source software for measuring and analyzing cell images
Exec=bash -c '\''export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 && cd "'"$CURRENT_DIR"'" && source venv_cellprofiler/bin/activate && python3 -m cellprofiler'\''
Icon=applications-science
Terminal=false
Type=Application
Categories=Science;Education;Graphics;
Keywords=cell;biology;image;analysis;microscopy;
StartupNotify=true'

create_desktop_file "cellprofiler.desktop" "$CELLPROFILER_CONTENT"

# CellProfiler Analyst .desktop file
CELLPROFILER_ANALYST_CONTENT='[Desktop Entry]
Name=CellProfiler Analyst
Comment=Interactive data exploration and analysis for CellProfiler measurements
Exec=bash -c '\''cd "'"$CURRENT_DIR"'" && source venv_cellprofileranalyst/bin/activate && python CellProfiler-Analyst-3.0.5/CellProfiler-Analyst.py'\''
Icon=applications-science
Terminal=false
Type=Application
Categories=Science;Education;Graphics;
Keywords=cell;biology;data;analysis;visualization;
StartupNotify=true'

create_desktop_file "cellprofiler-analyst.desktop" "$CELLPROFILER_ANALYST_CONTENT"

echo ""
echo ".desktop file generation complete."
echo "Desktop files are located in: $DESKTOP_DIR"
echo ""
echo "Double-click the desktop icons to launch the applications."
echo "Note: Seafront is configured to use 'squid' microscope. Edit seafront.desktop to change."
