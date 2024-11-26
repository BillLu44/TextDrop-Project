#!/usr/bin/bash

echo "Started client_script"
#echo "Starting NetworkManager"
#sudo systemctl start NetworkManager.service
#sudo nmcli device wifi hotspot con-name Group18 ssid Group18 band bg password qw3rtyu1
echo "Starting ssh"
sshpass -p "se101" ssh Group18@raspberrypi.local << EOF
    echo "setup ssh"
    cd python_scripts
    source pyvenv/bin/activate
    echo "activated environment"
    echo "attempt at server start"
    python3 server.py
    echo "server is up"
EOF