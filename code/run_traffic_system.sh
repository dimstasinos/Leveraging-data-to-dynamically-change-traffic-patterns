
if [ "$1" != "0" ]; then
    echo "Running dataset creation..."
    python3 dataset_creation.py

    echo "Running global scaler..."
    python3 global_scaler.py
else
    echo "Skipping dataset creation and global scaler as '0' is provided."
fi

# Start the traffic server in a new terminal
echo "Starting traffic server in a new terminal..."
gnome-terminal -- bash -c "python3 traffic_server.py; exec bash"

# Start the traffic client in a new terminal
echo "Starting traffic client in a new terminal..."
gnome-terminal -- bash -c "python3 traffic_client.py; exec bash"

echo "Both server and client are running in separate terminals."
