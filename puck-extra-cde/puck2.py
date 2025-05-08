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
                udp_ip="127.0.0.1",  # Changed default to localhost  
                udp_port=3001,
                calibrate_on_start=True,
                enable_udp=False):  # Added option to disable UDP
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
        self.enable_udp = enable_udp  # Control whether UDP is active
        
        # Puck detection settings for red puck (HSV color space)
        # Red is tricky in HSV as it wraps around the hue spectrum
        # We'll use two ranges and combine them
        self.red_lower1 = np.array([0, 120, 70])     # Lower range 1 for red
        self.red_upper1 = np.array([10, 255, 255])   # Upper range 1 for red
        self.red_lower2 = np.array([160, 120, 70])   # Lower range 2 for red
        self.red_upper2 = np.array([180, 255, 255])  # Upper range 2 for red
        
        # Tracking variables
        self.puck_position_px = None  # Position in pixels (x, y)
        self.puck_position_mm = None  # Position in mm (x, y) after calibration
        self.puck_radius_px = 0
        self.puck_detected = False
        self.last_detection_time = 0
        self.puck_velocity = (0, 0)   # Estimated velocity (vx, vy) in pixels/sec
        self.prev_position = None
        self.prev_detection_time = 0
        
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
        
        # Debug mode
        self.debug_mode = True
        self.debug_screens = {}
        
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
            
            # Initialize UDP socket if enabled
            if self.enable_udp:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                print(f"UDP socket initialized for sending to {self.udp_ip}:{self.udp_port}")
            else:
                print("UDP sending disabled")
            
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
        """Interactive calibration to set HSV color thresholds for red puck detection"""
        if self.camera is None:
            print("Camera not initialized")
            return False
        
        print("\n=== Red Puck Color Calibration ===")
        print("Adjust sliders to detect the red puck")
        print("Press 'S' to save and exit, 'ESC' to cancel")
        
        # Function for trackbar callbacks
        def nothing(x):
            pass
        
        # Create trackbar window
        cv2.namedWindow("Puck Color Calibration")
        
        # Create trackbars for both red ranges
        # First red range (0-10)
        cv2.createTrackbar("H Min1", "Puck Color Calibration", int(self.red_lower1[0]), 179, nothing)
        cv2.createTrackbar("H Max1", "Puck Color Calibration", int(self.red_upper1[0]), 179, nothing)
        
        # Second red range (160-180)
        cv2.createTrackbar("H Min2", "Puck Color Calibration", int(self.red_lower2[0]), 179, nothing)
        cv2.createTrackbar("H Max2", "Puck Color Calibration", int(self.red_upper2[0]), 179, nothing)
        
        # Common S and V ranges for both
        cv2.createTrackbar("S Min", "Puck Color Calibration", int(self.red_lower1[1]), 255, nothing)
        cv2.createTrackbar("S Max", "Puck Color Calibration", int(self.red_upper1[1]), 255, nothing)
        cv2.createTrackbar("V Min", "Puck Color Calibration", int(self.red_lower1[2]), 255, nothing)
        cv2.createTrackbar("V Max", "Puck Color Calibration", int(self.red_upper1[2]), 255, nothing)
        
        while self.camera.isOpened():
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to capture frame for color calibration")
                break
            
            # Convert to HSV
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Get current trackbar positions
            h_min1 = cv2.getTrackbarPos("H Min1", "Puck Color Calibration")
            h_max1 = cv2.getTrackbarPos("H Max1", "Puck Color Calibration")
            h_min2 = cv2.getTrackbarPos("H Min2", "Puck Color Calibration")
            h_max2 = cv2.getTrackbarPos("H Max2", "Puck Color Calibration")
            s_min = cv2.getTrackbarPos("S Min", "Puck Color Calibration")
            s_max = cv2.getTrackbarPos("S Max", "Puck Color Calibration")
            v_min = cv2.getTrackbarPos("V Min", "Puck Color Calibration")
            v_max = cv2.getTrackbarPos("V Max", "Puck Color Calibration")
            
            # Create both red masks
            lower_bound1 = np.array([h_min1, s_min, v_min])
            upper_bound1 = np.array([h_max1, s_max, v_max])
            mask1 = cv2.inRange(hsv_frame, lower_bound1, upper_bound1)
            
            lower_bound2 = np.array([h_min2, s_min, v_min])
            upper_bound2 = np.array([h_max2, s_max, v_max])
            mask2 = cv2.inRange(hsv_frame, lower_bound2, upper_bound2)
            
            # Combine masks
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Apply noise reduction
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
                self.red_lower1 = np.array([h_min1, s_min, v_min])
                self.red_upper1 = np.array([h_max1, s_max, v_max])
                self.red_lower2 = np.array([h_min2, s_min, v_min])
                self.red_upper2 = np.array([h_max2, s_max, v_max])
                print(f"Color settings saved: ")
                print(f"Lower1={self.red_lower1}, Upper1={self.red_upper1}")
                print(f"Lower2={self.red_lower2}, Upper2={self.red_upper2}")
                break
        
        cv2.destroyWindow("Puck Color Calibration")
        return True

    def detect_puck(self, frame):
        """Detect red puck in the given frame using improved detection algorithm"""
        if frame is None:
            return False
        
        # Make a copy for visualization
        vis_frame = frame.copy()
        
        # Convert to HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for both red ranges
        mask1 = cv2.inRange(hsv_frame, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv_frame, self.red_lower2, self.red_upper2)
        
        # Combine masks
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Apply Gaussian blur to reduce noise further
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Store debug information
        if self.debug_mode:
            self.debug_screens["hsv"] = cv2.cvtColor(hsv_frame, cv2.COLOR_HSV2BGR)
            self.debug_screens["mask"] = cv2.cvtColor(cv2.merge([mask, mask, mask]), cv2.COLOR_BGR2RGB)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Reset detection state
        self.puck_detected = False
        
        if contours:
            # Sort contours by area (largest first)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Filter contours by aspect ratio and area
            valid_contours = []
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Skip tiny contours
                if area < 100:
                    continue
                
                # Check if contour is approximately circular
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # For a perfect circle, circularity = 1
                # Accept values that are reasonably close to circular
                if circularity > 0.7:
                    valid_contours.append(contour)
            
            # Process the best contour if we found any valid ones
            if valid_contours:
                best_contour = valid_contours[0]  # Take the largest valid contour
                area = cv2.contourArea(best_contour)
                
                # Fit circle to contour
                (x, y), radius = cv2.minEnclosingCircle(best_contour)
                center = (int(x), int(y))
                radius = int(radius)
                
                # Calculate center using moments for better accuracy
                M = cv2.moments(best_contour)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    center = (cx, cy)
                
                # Store previous position for velocity calculation
                if self.puck_position_px:
                    self.prev_position = self.puck_position_px
                    self.prev_detection_time = self.last_detection_time
                
                # Update puck position
                self.puck_position_px = center
                self.puck_radius_px = radius
                self.puck_detected = True
                current_time = time.time()
                self.last_detection_time = current_time
                
                # Calculate velocity if we have previous position
                if self.prev_position and self.prev_detection_time > 0:
                    time_diff = current_time - self.prev_detection_time
                    if time_diff > 0:
                        dx = center[0] - self.prev_position[0]
                        dy = center[1] - self.prev_position[1]
                        
                        # Calculate velocity in pixels per second
                        vx = dx / time_diff
                        vy = dy / time_diff
                        
                        # Apply simple low-pass filter for smoothing
                        alpha = 0.7  # Weight for new measurement
                        self.puck_velocity = (
                            alpha * vx + (1 - alpha) * self.puck_velocity[0],
                            alpha * vy + (1 - alpha) * self.puck_velocity[1]
                        )
                
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
                cv2.circle(vis_frame, center, radius, (0, 255, 0), 2)
                cv2.circle(vis_frame, center, 2, (0, 0, 255), 3)
                
                # Draw velocity vector
                if self.puck_velocity != (0, 0):
                    # Scale velocity for visualization
                    scale = 2.0
                    end_point = (
                        int(center[0] + self.puck_velocity[0] * scale),
                        int(center[1] + self.puck_velocity[1] * scale)
                    )
                    cv2.arrowedLine(vis_frame, center, end_point, (255, 0, 0), 2)
                
                # Add position text
                if self.puck_position_mm:
                    cv2.putText(vis_frame, f"Pos(mm): ({self.puck_position_mm[0]:.1f}, {self.puck_position_mm[1]:.1f})",
                              (center[0] + 10, center[1] + 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    cv2.putText(vis_frame, f"Pos(px): {center[0]}, {center[1]}",
                              (center[0] + 10, center[1] + 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Add velocity text
                speed = np.sqrt(self.puck_velocity[0]**2 + self.puck_velocity[1]**2)
                cv2.putText(vis_frame, f"Speed: {speed:.1f} px/s",
                          (center[0] + 10, center[1] + 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Store debug information
                if self.debug_mode:
                    contour_img = np.zeros_like(frame)
                    cv2.drawContours(contour_img, [best_contour], 0, (0, 255, 0), 2)
                    self.debug_screens["contours"] = contour_img
        
        # Update debug screens
        if self.debug_mode:
            self.debug_screens["detection"] = vis_frame
        
        return self.puck_detected
    
    def send_position_data(self):
        """Send puck position data over UDP"""
        if not self.puck_detected or not self.enable_udp or self.socket is None:
            return False
        
        try:
            # Create data packet with position and velocity information
            data = {
                "detected": True,
                "timestamp": time.time(),
                "position_px": self.puck_position_px,
                "radius_px": self.puck_radius_px,
                "velocity_px": self.puck_velocity,
                "position_mm": self.puck_position_mm if self.puck_position_mm else (0, 0)
            }
            
            # Serialize position data and send
            # Format: "x,y,vx,vy" (all values in mm if calibrated)
            if self.puck_position_mm:
                # TODO: Convert velocity to mm/s when calibration is complete
                message = f"{data['position_mm'][0]:.2f},{data['position_mm'][1]:.2f},{self.puck_velocity[0]:.2f},{self.puck_velocity[1]:.2f}".encode('utf-8')
            else:
                # Send pixel coordinates if not calibrated
                message = f"{data['position_px'][0]:.2f},{data['position_px'][1]:.2f},{self.puck_velocity[0]:.2f},{self.puck_velocity[1]:.2f}".encode('utf-8')
            
            self.socket.sendto(message, (self.udp_ip, self.udp_port))
            return True
            
        except Exception as e:
            print(f"Error sending position data: {e}")
            return False
    
    def show_debug_screen(self):
        """Display debug screens for visual feedback"""
        if not self.debug_mode or not self.debug_screens:
            return
        
        # Create composite debug view
        debug_rows = []
        
        # First row: Original + HSV + Mask
        if "hsv" in self.debug_screens and "mask" in self.debug_screens:
            original = self.frame.copy() if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            first_row = np.hstack([
                original,
                self.debug_screens.get("hsv", np.zeros_like(original)),
                self.debug_screens.get("mask", np.zeros_like(original))
            ])
            debug_rows.append(first_row)
        
        # Second row: Detection + Contours + Empty
        if "detection" in self.debug_screens:
            detection = self.debug_screens["detection"]
            contours = self.debug_screens.get("contours", np.zeros_like(detection))
            # Add empty or additional debug screen
            empty = np.zeros_like(detection)
            cv2.putText(empty, "Debug Information", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add text with color range values
            cv2.putText(empty, f"Red Range 1: H({self.red_lower1[0]}-{self.red_upper1[0]})", 
                      (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(empty, f"Red Range 2: H({self.red_lower2[0]}-{self.red_upper2[0]})", 
                      (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(empty, f"S: {self.red_lower1[1]}-{self.red_upper1[1]}", 
                      (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(empty, f"V: {self.red_lower1[2]}-{self.red_upper1[2]}", 
                      (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.puck_detected:
                cv2.putText(empty, f"Puck Position: {self.puck_position_px}", 
                          (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(empty, f"Velocity: {self.puck_velocity[0]:.1f}, {self.puck_velocity[1]:.1f} px/s", 
                          (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                speed = np.sqrt(self.puck_velocity[0]**2 + self.puck_velocity[1]**2)
                cv2.putText(empty, f"Speed: {speed:.1f} px/s", 
                          (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(empty, "Puck not detected", 
                          (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            second_row = np.hstack([detection, contours, empty])
            debug_rows.append(second_row)
        
        # Stack rows vertically
        if debug_rows:
            debug_composite = np.vstack(debug_rows)
            
            # Resize if too large for screen
            screen_height = 900  # Max height
            if debug_composite.shape[0] > screen_height:
                scale = screen_height / debug_composite.shape[0]
                debug_composite = cv2.resize(debug_composite, 
                                          (int(debug_composite.shape[1] * scale), 
                                           int(debug_composite.shape[0] * scale)))
            
            cv2.imshow("Puck Tracker Debug", debug_composite)
            
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
            
            # Store original frame
            self.frame = frame.copy()
            
            # Detect puck
            puck_found = self.detect_puck(frame)
            
            # Send position data if puck detected and UDP is enabled
            if puck_found and self.enable_udp:
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
                vis_frame = self.debug_screens.get("detection", frame)
                
                # Draw table outline
                for i in range(4):
                    p1 = self.calibration_points_px[i]
                    p2 = self.calibration_points_px[(i+1) % 4]
                    cv2.line(vis_frame, p1, p2, (0, 0, 255), 2)
                
                # Add grid lines
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
                        
                        cv2.line(vis_frame, top_px, bottom_px, (100, 100, 100), 1)
                    
                    # Draw horizontal grid lines
                    for y_mm in range(0, int(self.table_height_mm) + 1, grid_size_mm):
                        # Convert left and right points from mm to pixels
                        left_mm = np.array([[[0, y_mm]]], dtype=np.float32)
                        right_mm = np.array([[[self.table_width_mm, y_mm]]], dtype=np.float32)
                        
                        left_px = cv2.perspectiveTransform(left_mm, inv_homography)[0][0]
                        right_px = cv2.perspectiveTransform(right_mm, inv_homography)[0][0]
                        
                        left_px = tuple(map(int, left_px))
                        right_px = tuple(map(int, right_px))
                        
                        cv2.line(vis_frame, left_px, right_px, (100, 100, 100), 1)
                
                self.debug_screens["detection"] = vis_frame
            
            # Add status text to main display
            main_display = self.debug_screens.get("detection", frame)
            cv2.putText(main_display, f"FPS: {self.fps:.1f}", (10, 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(main_display, "Puck: " + ("Detected" if self.puck_detected else "Not detected"), 
                      (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                      (0, 255, 0) if self.puck_detected else (0, 0, 255), 1)
            
            if self.puck_position_mm:
                cv2.putText(main_display, f"Position (mm): ({self.puck_position_mm[0]:.1f}, {self.puck_position_mm[1]:.1f})", 
                          (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the main window and debug screens
            cv2.imshow("Air Hockey Puck Tracker", main_display)
            self.show_debug_screen()
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.running = False
                break
            elif key == ord('d'):
                # Toggle debug mode
                self.debug_mode = not self.debug_mode
                if not self.debug_mode:
                    cv2.destroyWindow("Puck Tracker Debug")
            elif key == ord('c'):
                # Toggle color calibration
                self.calibrate_puck_color()
            elif key == ord('t'):
                # Toggle table calibration
                self.calibrate_table()
    
    def start(self):
        """Start tracking in a separate thread"""
        if self.camera is None:
            print("Camera not initialized. Call initialize() first.")
            return False
        
        self.running = True
        self.processing_thread = threading.Thread(target=self.process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("\nPuck tracker started")
        print("--------------------")
        print("Press 'q' to quit")
        print("Press 'd' to toggle debug view")
        print("Press 'c' to recalibrate puck color")
        print("Press 't' to recalibrate table")
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
    # Create tracker with UDP disabled by default
    tracker = PuckTracker(
        camera_id=0,           # Use first camera
        udp_ip="127.0.0.1",    # Default to localhost 
        udp_port=3001,         # Default port
        enable_udp=False       # Disable UDP by default
    )
    
    try:
        # Initialize camera and UDP (if enabled)
        if not tracker.initialize():
            print("Failed to initialize tracker")
            return
        
        # Start processing
        if not tracker.start():
            print("Failed to start tracking")
            return
        
        # Wait for user to quit
        while tracker.running:
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        tracker.cleanup()

if __name__ == "__main__":
    main()