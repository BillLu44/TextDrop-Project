#!/usr/bin/bash

echo "Started client_script"
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