import cv2
import numpy as np
import socket
import struct
import time
import threading
from typing import Tuple, Optional, List

class PuckTracker:
    def __init__(self, 
                camera_id=0, 
                 udp_ip="192.168.4.201", 
                 udp_port=3001,
                 calibrate_on_start=True):
        # Camera settings
        self.camera_id = camera_id
        self.camera = None
        self.frame = None
        self.frame_width = 0
        self.frame_height = 0
        
        # UDP settings for sending coordinates
        self.udp_ip = udp_ip
        self.udp_port = udp_port
        self.socket = None
        
        # Puck detection settings (default: orange puck in HSV color space)
        self.puck_color_lower = np.array([0, 100, 100])  # Orange-red in HSV
        self.puck_color_upper = np.array([15, 255, 255])
        
        # Tracking variables
        self.puck_position_px = None  # Position in pixels (x, y)
        self.puck_position_mm = None  # Position in mm (x, y) after calibration
        self.puck_radius_px = 0
        self.puck_detected = False
        self.last_detection_time = 0
        
        # Table calibration
        self.calibration_points_px = []  # Pixel coordinates of table corners
        self.calibration_done = False
        self.homography_matrix = None   # For pixel to mm conversion
        
        # Table dimensions in mm (standard air hockey table)
        self.table_width_mm = 1600  # Adjust to your actual table dimensions
        self.table_height_mm = 800
        
        # Running control
        self.running = False
        self.processing_thread = None
        
        # Statistics
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = 0
        
        # Auto-calibrate at start if requested
        self.calibrate_on_start = calibrate_on_start
    
    def initialize(self):
        """Initialize camera and UDP socket"""
        try:
            # Initialize camera
            self.camera = cv2.VideoCapture(self.camera_id)
            if not self.camera.isOpened():
                raise Exception(f"Failed to open camera with ID {self.camera_id}")
            
            # Get camera resolution
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
            
            # Set camera properties for better performance if needed
            self.camera.set(cv2.CAP_PROP_FPS, 60)  # Try to set higher FPS if camera supports it
            
            # Initialize UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            print(f"UDP socket initialized for sending to {self.udp_ip}:{self.udp_port}")
            
            # Perform calibration if requested
            if self.calibrate_on_start:
                self.calibrate_table()
                self.calibrate_puck_color()
            
            return True
        except Exception as e:
            print(f"Initialization error: {e}")
            self.cleanup()
            return False
    
    def calibrate_table(self):
        """Interactive calibration to map between camera pixels and real-world coordinates"""
        if self.camera is None:
            print("Camera not initialized")
            return False
        
        print("\n=== Table Calibration ===")
        print("Click on the four corners of the table in this order:")
        print("1. Top-left")
        print("2. Top-right")
        print("3. Bottom-right")
        print("4. Bottom-left")
        
        # Reset calibration points
        self.calibration_points_px = []
        
        # Capture a frame for calibration
        ret, frame = self.camera.read()
        if not ret:
            print("Failed to capture frame for calibration")
            return False
        
        calibration_frame = frame.copy()
        
        # Setup mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point
                self.calibration_points_px.append((x, y))
                
                # Draw point
                cv2.circle(calibration_frame, (x, y), 5, (0, 255, 0), -1)
                
                # Draw lines between points
                if len(self.calibration_points_px) > 1:
                    for i in range(1, len(self.calibration_points_px)):
                        cv2.line(calibration_frame, 
                                self.calibration_points_px[i-1], 
                                self.calibration_points_px[i], 
                                (0, 255, 0), 2)
                
                # Connect last point to first if we have all four
                if len(self.calibration_points_px) == 4:
                    cv2.line(calibration_frame, 
                            self.calibration_points_px[3], 
                            self.calibration_points_px[0], 
                            (0, 255, 0), 2)
                    
                    # Show completion message
                    cv2.putText(calibration_frame, "Calibration complete! Press any key to continue.",
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Update display
                cv2.imshow("Table Calibration", calibration_frame)
        
        # Setup window and callback
        cv2.namedWindow("Table Calibration")
        cv2.setMouseCallback("Table Calibration", mouse_callback)
        cv2.imshow("Table Calibration", calibration_frame)
        
        # Wait for calibration to complete
        cv2.waitKey(0)
        cv2.destroyWindow("Table Calibration")
        
        # Check if we got all four corners
        if len(self.calibration_points_px) != 4:
            print(f"Calibration incomplete: {len(self.calibration_points_px)}/4 points selected")
            return False
        
        # Convert to numpy array for homography calculation
        src_points = np.array(self.calibration_points_px, dtype=np.float32)
        
        # Define destination points (real-world coordinates in mm)
        dst_points = np.array([
            [0, 0],  # Top-left
            [self.table_width_mm, 0],  # Top-right
            [self.table_width_mm, self.table_height_mm],  # Bottom-right
            [0, self.table_height_mm]  # Bottom-left
        ], dtype=np.float32)
        
        # Calculate homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.calibration_done = True
        
        print("Table calibration complete")
        return True

    def calibrate_puck_color(self):
        """Interactive calibration to set HSV color thresholds for puck detection"""
        if self.camera is None:
            print("Camera not initialized")
            return False
        
        print("\n=== Puck Color Calibration ===")
        print("Adjust sliders to detect the puck")
        print("Press 'S' to save and exit, 'ESC' to cancel")
        
        # Function for trackbar callbacks
        def nothing(x):
            pass
        
        # Create trackbar window
        cv2.namedWindow("Puck Color Calibration")
        cv2.createTrackbar("H Min", "Puck Color Calibration", int(self.puck_color_lower[0]), 179, nothing)
        cv2.createTrackbar("H Max", "Puck Color Calibration", int(self.puck_color_upper[0]), 179, nothing)
        cv2.createTrackbar("S Min", "Puck Color Calibration", int(self.puck_color_lower[1]), 255, nothing)
        cv2.createTrackbar("S Max", "Puck Color Calibration", int(self.puck_color_upper[1]), 255, nothing)
        cv2.createTrackbar("V Min", "Puck Color Calibration", int(self.puck_color_lower[2]), 255, nothing)
        cv2.createTrackbar("V Max", "Puck Color Calibration", int(self.puck_color_upper[2]), 255, nothing)
        
        while self.camera.isOpened():
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame for color calibration")
                break
            
            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get current trackbar positions
            h_min = cv2.getTrackbarPos("H Min", "Puck Color Calibration")
            h_max = cv2.getTrackbarPos("H Max", "Puck Color Calibration")
            s_min = cv2.getTrackbarPos("S Min", "Puck Color Calibration")
            s_max = cv2.getTrackbarPos("S Max", "Puck Color Calibration")
            v_min = cv2.getTrackbarPos("V Min", "Puck Color Calibration")
            v_max = cv2.getTrackbarPos("V Max", "Puck Color Calibration")
            
            # Create color mask
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
            
            # Apply some filtering to the mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # Find contours in mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create result with mask applied
            result = cv2.bitwise_and(frame, frame, mask=mask)
            
            # Draw circles around potential pucks
            detection_frame = frame.copy()
            if contours:
                # Find the largest contour (presumably the puck)
                largest_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest_contour)
                
                # Only consider it a puck if the area is reasonable
                if area > 100:
                    # Fit circle to contour
                    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    # Draw the circle
                    cv2.circle(detection_frame, center, radius, (0, 255, 0), 2)
                    cv2.circle(detection_frame, center, 2, (0, 0, 255), 3)
                    
                    # Show position
                    cv2.putText(detection_frame, f"Position: ({center[0]}, {center[1]})",
                               (center[0] + 10, center[1] + 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Show area
                    cv2.putText(detection_frame, f"Area: {area:.0f}",
                               (center[0] + 10, center[1] + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Stack images for display
            stacked_images = np.hstack([frame, detection_frame, result])
            
            # Resize if too large for screen
            if stacked_images.shape[1] > 1600:
                scale = 1600 / stacked_images.shape[1]
                stacked_images = cv2.resize(stacked_images, 
                                         (int(stacked_images.shape[1] * scale), 
                                          int(stacked_images.shape[0] * scale)))
            
            # Show instructions
            cv2.putText(stacked_images, "Original | Detection | Mask",
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(stacked_images, "Press 'S' to save settings, 'ESC' to cancel",
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show result
            cv2.imshow("Puck Color Calibration", stacked_images)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                print("Color calibration cancelled")
                break
            elif key == ord('s'):  # 'S' key
                # Save color values
                self.puck_color_lower = np.array([h_min, s_min, v_min])
                self.puck_color_upper = np.array([h_max, s_max, v_max])
                print(f"Color settings saved: Lower={self.puck_color_lower}, Upper={self.puck_color_upper}")
                break
        
        cv2.destroyWindow("Puck Color Calibration")
        return True

    def detect_puck(self, frame):
        """Detect puck in the given frame"""
        if frame is None:
            return False
        
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask using color thresholds
        mask = cv2.inRange(hsv_frame, self.puck_color_lower, self.puck_color_upper)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset detection state
        self.puck_detected = False
        
        if contours:
            # Find the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Only consider it a puck if area is reasonable
            # Adjust these thresholds based on your puck size and camera distance
            min_area = 100
            max_area = 5000
            
            if min_area < area < max_area:
                # Get center and radius using minimum enclosing circle
                (x, y), radius = cv2.minEnclosingCircle(largest_contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Alternative: Use moments for better center calculation
                M = cv2.moments(largest_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                
                # Update puck position
                self.puck_position_px = center
                self.puck_radius_px = radius
                self.puck_detected = True
                self.last_detection_time = time.time()
                
                # Convert to real-world coordinates if calibration is done
                if self.calibration_done and self.homography_matrix is not None:
                    # Reshape for perspective transform
                    point = np.array([[[center[0], center[1]]]], dtype=np.float32)
                    transformed = cv2.perspectiveTransform(point, self.homography_matrix)
                    self.puck_position_mm = (
                        float(transformed[0][0][0]), 
                        float(transformed[0][0][1])
                    )
                
                # Draw circle on visualization frame
                cv2.circle(frame, center, radius, (0, 255, 0), 2)
                cv2.circle(frame, center, 2, (0, 0, 255), 3)
                
                # Add position text
                if self.puck_position_mm:
                    cv2.putText(frame, f"Pos(mm): ({self.puck_position_mm[0]:.1f}, {self.puck_position_mm[1]:.1f})",
                              (center[0] + 10, center[1] + 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(frame, f"Pos(px): {center[0]}, {center[1]}",
                              (center[0] + 10, center[1] + 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                return True
        
        return False
    
    def send_position_data(self):
        """Send puck position data over UDP"""
        if not self.puck_detected or self.socket is None:
            return False
        
        try:
            # Create data packet (send both pixel and mm coordinates if available)
            data = {
                "detected": True,
                "timestamp": time.time(),
                "position_px": self.puck_position_px,
                "radius_px": self.puck_radius_px,
                "position_mm": self.puck_position_mm if self.puck_position_mm else (0, 0)
            }
            
            # Serialize data to JSON and send
            message = f"{data['position_mm'][0]:.2f},{data['position_mm'][1]:.2f}".encode('utf-8')
            
            # For more complex data use UDP packet format from the spec
            # Here's how to send the position as a basic UDP message
            self.socket.sendto(message, (self.udp_ip, self.udp_port))
            return True
            
        except Exception as e:
            print(f"Error sending position data: {e}")
            return False
            
    def process_frames(self):
        """Main processing loop"""
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        while self.running and self.camera.isOpened():
            # Capture new frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Process frame
            self.frame = frame.copy()
            
            # Detect puck
            puck_found = self.detect_puck(frame)
            
            # Send position data if puck detected
            if puck_found:
                self.send_position_data()
            
            # Update FPS counter
            self.frame_count += 1
            current_time = time.time()
            time_diff = current_time - self.last_fps_time
            
            if time_diff >= 1.0:
                self.fps = self.frame_count / time_diff
                self.frame_count = 0
                self.last_fps_time = current_time
                print(f"FPS: {self.fps:.1f}")
            
            # Display calibration grid if calibration is done
            if self.calibration_done and len(self.calibration_points_px) == 4:
                # Draw table outline
                for i in range(4):
                    p1 = self.calibration_points_px[i]
                    p2 = self.calibration_points_px[(i+1) % 4]
                    cv2.line(frame, p1, p2, (0, 0, 255), 2)
                
                # Add grid lines (optional)
                # This draws a 10x10 grid on the table
                if self.homography_matrix is not None:
                    grid_size_mm = 100  # mm between grid lines
                    
                    # Inverse homography to go from mm to pixels
                    inv_homography = np.linalg.inv(self.homography_matrix)
                    
                    # Draw vertical grid lines
                    for x_mm in range(0, int(self.table_width_mm) + 1, grid_size_mm):
                        # Convert top and bottom points from mm to pixels
                        top_mm = np.array([[[x_mm, 0]]], dtype=np.float32)
                        bottom_mm = np.array([[[x_mm, self.table_height_mm]]], dtype=np.float32)
                        
                        top_px = cv2.perspectiveTransform(top_mm, inv_homography)[0][0]
                        bottom_px = cv2.perspectiveTransform(bottom_mm, inv_homography)[0][0]
                        
                        top_px = tuple(map(int, top_px))
                        bottom_px = tuple(map(int, bottom_px))
                        
                        cv2.line(frame, top_px, bottom_px, (100, 100, 100), 1)
                    
                    # Draw horizontal grid lines
                    for y_mm in range(0, int(self.table_height_mm) + 1, grid_size_mm):
                        # Convert left and right points from mm to pixels
                        left_mm = np.array([[[0, y_mm]]], dtype=np.float32)
                        right_mm = np.array([[[self.table_width_mm, y_mm]]], dtype=np.float32)
                        
                        left_px = cv2.perspectiveTransform(left_mm, inv_homography)[0][0]
                        right_px = cv2.perspectiveTransform(right_mm, inv_homography)[0][0]
                        
                        left_px = tuple(map(int, left_px))
                        right_px = tuple(map(int, right_px))
                        
                        cv2.line(frame, left_px, right_px, (100, 100, 100), 1)
            
            # Add status text
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, "Puck: " + ("Detected" if self.puck_detected else "Not detected"), 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 255, 0) if self.puck_detected else (0, 0, 255), 1)
            
            if self.puck_position_mm:
                cv2.putText(frame, f"Position (mm): ({self.puck_position_mm[0]:.1f}, {self.puck_position_mm[1]:.1f})", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the frame
            cv2.imshow("Air Hockey Puck Tracker", frame)
            
            # Check for key press (q to quit)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
    
    def start(self):
        """Start tracking in a separate thread"""
        if self.camera is None:
            print("Camera not initialized. Call initialize() first.")
            return False
        
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        return True
    
    def stop(self):
        """Stop tracking"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop()
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        
        cv2.destroyAllWindows()
        print("Resources cleaned up")

# Example usage
def main():
    # Create tracker
    tracker = PuckTracker(
        camera_id=0,  # Use first camera
        udp_ip="192.168.4.201",  # PLC/gantry IP
        udp_port=3001  # PLC port for receiving setpoints
    )
    
    try:
        # Initialize camera and UDP
        if not tracker.initialize():
            print("Failed to initialize tracker")
            return
        
        # Start processing
        if not tracker.start():
            print("Failed to start tracking")
            return
        
        print("\nAir Hockey Puck Tracker running")
        print("--------------------------------")
        print("Press 'q' in the video window to quit")
        
        # Wait for user to quit
        while tracker.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()