#!/bin/bash

START_WD=$(pwd)

# Define the Python version
PYTHON_VERSION="3.10.9"

# Define the installation directories
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd $SCRIPT_DIR

PYTHON_INSTALL_DIR="$SCRIPT_DIR/python-$PYTHON_VERSION"
PYTHON_SOURCE_DIR="$SCRIPT_DIR/python-src"
OPENSSL_INSTALL_DIR="$SCRIPT_DIR/openssl"
OPENSSL_SOURCE_DIR="$SCRIPT_DIR/openssl-src"
XZ_INSTALL_DIR="$SCRIPT_DIR/xz"
XZ_SOURCE_DIR="$SCRIPT_DIR/xz-src"

# download and compile openssl for use by python
curl -o openssl.tar.gz https://www.openssl.org/source/openssl-3.2.1.tar.gz
mkdir -p $OPENSSL_INSTALL_DIR $OPENSSL_SOURCE_DIR
tar -xzf openssl.tar.gz -C $OPENSSL_SOURCE_DIR --strip-components=1
cd $OPENSSL_SOURCE_DIR
./config --prefix=$OPENSSL_INSTALL_DIR --openssldir=$OPENSSL_INSTALL_DIR/ssl
make -j
# make -j test
make -j install

# remove openssl source code files after installation
cd $SCRIPT_DIR
rm -rf $OPENSSL_SOURCE_DIR $SCRIPT_DIR/openssl.tar.gz

# download and compile xz to provide lzma support for python
curl -L -o xz.tar.gz https://github.com/tukaani-project/xz/releases/download/v5.4.6/xz-5.4.6.tar.gz # this is a post-backdoor version
mkdir -p $XZ_INSTALL_DIR $XZ_SOURCE_DIR
tar -xzf xz.tar.gz -C $XZ_SOURCE_DIR --strip-components=1
cd $XZ_SOURCE_DIR
./configure --prefix=$XZ_INSTALL_DIR
make -j
make -j install

# remove xz source code files after installation
cd $SCRIPT_DIR
rm -rf $XZ_SOURCE_DIR $SCRIPT_DIR/xz.tar.gz

# Update environment variables to include xz and openssl
export PATH=$XZ_INSTALL_DIR/bin:$OPENSSL_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$XZ_INSTALL_DIR/lib:$OPENSSL_INSTALL_DIR/lib:$LD_LIBRARY_PATH
export C_INCLUDE_PATH=$XZ_INSTALL_DIR/include:$OPENSSL_INSTALL_DIR/include:$C_INCLUDE_PATH
export LIBRARY_PATH=$XZ_INSTALL_DIR/lib:$OPENSSL_INSTALL_DIR/lib:$LIBRARY_PATH
export LDFLAGS="${LDFLAGS} -Wl,-rpath=$OPENSSL_INSTALL_DIR/lib"

# Download and compile python
PYTHON_URL="https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz"
curl -o python.tgz $PYTHON_URL
mkdir -p $PYTHON_INSTALL_DIR $PYTHON_SOURCE_DIR
tar -xzf python.tgz -C $PYTHON_SOURCE_DIR --strip-components=1
cd $PYTHON_SOURCE_DIR
# Configure and make
# + enable ssl support (required by pip for pypi packages) - this requires openssl to be installed on the system!
# + also apply several optimizations to improve runtime performance (no --enable-optimizations flag \
#   because pgo generates wrong raw profile data, version=8 instead of expected 9?!)
./configure --prefix=$PYTHON_INSTALL_DIR --with-openssl=$OPENSSL_INSTALL_DIR --with-openssl-rpath=auto --with-lto --with-computed-gotos --with-ensurepip
make -j
make -j install

# remove python source code files after installation
cd $SCRIPT_DIR
rm -rf $PYTHON_SOURCE_DIR $SCRIPT_DIR/python.tgz

echo "Python $PYTHON_VERSION installed successfully to $PYTHON_INSTALL_DIR"
echo "Use 'source activate.sh' to activate this Python environment."

# install package dependencies
cd $SCRIPT_DIR
source activate.sh
python3 -m pip install --upgrade pip

cd $SCRIPT_DIR/.. # project root dir
python3 -m pip install -e .
