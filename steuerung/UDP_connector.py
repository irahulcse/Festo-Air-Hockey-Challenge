import socket
import struct

class UDPConnector():
    def init__(self, host, port):
        # Define PLC network configuration
        PLC_IP = '192.168.4.201'
        PLC_PORT = 3001

        # Create a UDP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        self.enable, self.acknowledge, self.velocity, self.acceleration = self.define_setpoint()
        

    def define_setpoints(self, enable = True, acknowledge = False, velocity= 0.0 , acceleration= 0.0) :
        # Define the setpoint values
        # Pack the data into 40 bytes using little-endian format
        # Format: <BB6xdddd (2 booleans, 6 bytes padding, 4 doubles)
        return (enable, acknowledge, velocity, acceleration)
    
         
    def send_coordinates(self,x, y):
        message = struct.pack('<BB6xdddd', self.enable, self.acknowledge, self.velocity, self.acceleration, x, y)
        # Send the UDP message to the PLC
        self.sock.sendto(message, (self.PLC_IP, self.PLC_PORT))
        print("UDP setpoint (0.0, 0.0) sent to PLC.")
