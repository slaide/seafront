git clone https://github.com/slaide/toupcam-sdk --depth 1
cd toupcam-sdk/toupcam/lib/udev
sudo cp 99-toupcam.rules /etc/udev/rules.d/
cd ../../../..
rm -rf toupcam-sdk
