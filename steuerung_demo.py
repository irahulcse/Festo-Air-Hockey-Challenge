import socket
import struct

# Define PLC network configuration
PLC_IP = '192.168.4.201'
PLC_PORT = 3001

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Define the setpoint values
enable = True           # Start motion
acknowledge = False     # No error acknowledgment
velocity = 0.0          # 0 = max velocity
acceleration = 0.0      # 0 = max acceleration
x = 0.0                 # Target X position in mm
y = 0.0                 # Target Y position in mm

# Pack the data into 40 bytes using little-endian format
# Format: <BB6xdddd (2 booleans, 6 bytes padding, 4 doubles)
message = struct.pack('<BB6xdddd', enable, acknowledge, velocity, acceleration, x, y)

# Send the UDP message to the PLC
sock.sendto(message, (PLC_IP, PLC_PORT))
print("UDP setpoint (0.0, 0.0) sent to PLC.")
