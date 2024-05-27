wget  https://www.pjrc.com/teensy/00-teensy.rules
sudo cp 00-teensy.rules /etc/udev/rules.d/
rm -f 00-teensy.rules
echo "microcontroller udev rules installed"