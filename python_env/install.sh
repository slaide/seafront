if [ ! -e python.makefile ] ; then
curl -SsL -o python.makefile https://raw.githubusercontent.com/slaide/linux-snippets/refs/heads/main/python.makefile
fi
make -f python.makefile all PYTHON_VERSION=3.13.0 -j -l3
chmod +x python-3.13.0/bin/python3
