
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import math
import os

class DebugLungeCounter:
    def __init__(self, model_name='yolo11s-pose.pt'):
        # Load YOLO11 pose model
        print(f"Loading {model_name}...")
        self.model = YOLO(model_name)
        
        # Lunge counting parameters
        self.lunge_count = 0
        self.lunge_stage = "up"  # "up" or "down"
        self.confidence_threshold = 0.3
        
        # UPDATED Knee angle thresholds based on your analysis
        self.knee_down_threshold = 135  # Degrees - bent knee (lunge down) - INCREASED
        self.knee_up_threshold = 155    # Degrees - straight knee (lunge up) - INCREASED
        
        # For smoothing
        self.knee_angles_history = deque(maxlen=5)
        
        # Debug counters
        self.frame_count = 0
        self.pose_detected_count = 0
        self.knee_detected_count = 0
        
        # print(f"YOLO11 model loaded with UPDATED thresholds")
        # print(f"NEW thresholds: Down < {self.knee_down_threshold}°, Up > {self.knee_up_threshold}°")
        
    def detect_pose_yolo11(self, frame):
        """Detect pose using YOLO11 with debugging."""
        results = self.model(frame)
        
        if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
            keypoints = results[0].keypoints.xy[0].cpu().numpy()
            confidences = results[0].keypoints.conf[0].cpu().numpy() if results[0].keypoints.conf is not None else None
            
            self.pose_detected_count += 1
            if self.frame_count % 100 == 0:  # Debug every 100 frames
                print(f" Frame {self.frame_count}: Pose detected, {len(keypoints)} keypoints")
            
            return keypoints, confidences, results[0]
        
        if self.frame_count % 100 == 0:
            print(f"Frame {self.frame_count}: No pose detected")
        
        return None, None, results[0]
    
    def calculate_angle(self, point1, point2, point3):
        """Calculate angle between three points with debugging."""
        try:
            a = np.array(point1)
            b = np.array(point2)
            c = np.array(point3)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
            
            angle = np.arccos(cosine_angle)
            angle_degrees = np.degrees(angle)
            
            return angle_degrees
        
        except Exception as e:
            if self.frame_count % 200 == 0:
                print(f" Angle calculation error: {e}")
            return 180
    
    def get_knee_angles(self, keypoints, confidences):
        """Get knee angles with CORRECTED indexing and detailed debugging."""
        if keypoints is None or len(keypoints) < 17:
            if self.frame_count % 100 == 0:
                print(f"Frame {self.frame_count}: Insufficient keypoints ({len(keypoints) if keypoints is not None else 0})")
            return None, None, None, None
        
        # CORRECTED YOLO11 keypoint indices (0-indexed)
        # Left leg: hip(11) -> knee(13) -> ankle(15)
        # Right leg: hip(12) -> knee(14) -> ankle(16)
        
        left_hip = keypoints[11]      # FIXED: was keypoints
        left_knee = keypoints[13]     # FIXED: was keypoints  
        left_ankle = keypoints[15]    # FIXED: was keypoints
        
        right_hip = keypoints[12]     # FIXED: was keypoints
        right_knee = keypoints[14]    # FIXED: was keypoints
        right_ankle = keypoints[16]   # FIXED: was keypoints
        
        # Check confidence scores
        if confidences is not None:
            left_conf = min(confidences[11], confidences[13], confidences[15])   # FIXED: was confidences
            right_conf = min(confidences[12], confidences[14], confidences[16])  # FIXED: was confidences
        else:
            left_conf = 1.0
            right_conf = 1.0
        
        left_angle = None
        right_angle = None
        
        # Debug keypoint positions
        if self.frame_count % 100 == 0:
            print(f"Frame {self.frame_count} Keypoints:")
            print(f"   Left: Hip{left_hip}, Knee{left_knee}, Ankle{left_ankle}, Conf:{left_conf:.2f}")
            print(f"   Right: Hip{right_hip}, Knee{right_knee}, Ankle{right_ankle}, Conf:{right_conf:.2f}")
        
        # Calculate left knee angle - FIXED coordinate access
        if (left_hip[0] > 0 and left_hip[1] > 0 and 
            left_knee[0] > 0 and left_knee[1] > 0 and 
            left_ankle[0] > 0 and left_ankle[1] > 0 and 
            left_conf > self.confidence_threshold):
            
            left_angle = self.calculate_angle(left_hip, left_knee, left_ankle)
            self.knee_detected_count += 1
            
        # Calculate right knee angle - FIXED coordinate access
        if (right_hip[0] > 0 and right_hip[1] > 0 and 
            right_knee[0] > 0 and right_knee[1] > 0 and 
            right_ankle[0] > 0 and right_ankle[1] > 0 and 
            right_conf > self.confidence_threshold):
            
            right_angle = self.calculate_angle(right_hip, right_knee, right_ankle)
            self.knee_detected_count += 1
        
        # Debug angle values
        if self.frame_count % 50 == 0 and (left_angle is not None or right_angle is not None):
            print(f" Frame {self.frame_count} Angles: Left={left_angle:.1f}° Right={right_angle:.1f}°")
            print(f"   Thresholds: Down<{self.knee_down_threshold}°, Up>{self.knee_up_threshold}°")
        
        return left_angle, right_angle, left_conf, right_conf
    
    def update_lunge_count(self, left_angle, right_angle):
        """Update lunge count with UPDATED thresholds and detailed debugging."""
        
        # Determine working leg
        working_angle = None
        working_leg = None
        
        if left_angle is not None and right_angle is not None:
            if left_angle < right_angle:
                working_angle = left_angle
                working_leg = "left"
            else:
                working_angle = right_angle
                working_leg = "right"
        elif left_angle is not None:
            working_angle = left_angle
            working_leg = "left"
        elif right_angle is not None:
            working_angle = right_angle
            working_leg = "right"
        
        if working_angle is None:
            if self.frame_count % 100 == 0:
                print(f"Frame {self.frame_count}: No working angle available")
            return False, None
        
        # Smooth the angle
        self.knee_angles_history.append(working_angle)
        if len(self.knee_angles_history) >= 3:
            smoothed_angle = np.mean(list(self.knee_angles_history))
        else:
            smoothed_angle = working_angle
        
        # Debug angle and stage
        if self.frame_count % 30 == 0:
            print(f"Frame {self.frame_count}: {working_leg} leg, Raw={working_angle:.1f}°, Smooth={smoothed_angle:.1f}°, Stage={self.lunge_stage}")
        
        # Lunge counting logic with UPDATED thresholds
        lunge_detected = False
        old_stage = self.lunge_stage
        
        if smoothed_angle < self.knee_down_threshold and self.lunge_stage == "up":
            self.lunge_stage = "down"
            print(f"Frame {self.frame_count}: {working_leg} leg DOWN - angle {smoothed_angle:.1f}° < {self.knee_down_threshold}°")
            
        elif smoothed_angle > self.knee_up_threshold and self.lunge_stage == "down":
            self.lunge_stage = "up"
            self.lunge_count += 1
            lunge_detected = True
            print(f"Frame {self.frame_count}: {working_leg} leg UP - angle {smoothed_angle:.1f}° > {self.knee_up_threshold}°")
            print(f"LUNGE COMPLETED! Total count: {self.lunge_count}")
        
        return lunge_detected, working_leg
    
    def get_count(self):
        return self.lunge_count
    
    def get_stage(self):
        return self.lunge_stage
    
    def print_debug_summary(self):
        """Print debugging summary."""
        print(f"\nDEBUG SUMMARY after {self.frame_count} frames:")
        print(f"   Pose detected: {self.pose_detected_count}/{self.frame_count} ({(self.pose_detected_count/self.frame_count)*100:.1f}%)")
        print(f"   Knee detected: {self.knee_detected_count}")
        print(f"   Lunges counted: {self.lunge_count}")
        print(f"   Current stage: {self.lunge_stage}")
        print(f"   Thresholds: Down<{self.knee_down_threshold}°, Up>{self.knee_up_threshold}°")


def process_debug_lunge_video(video_path, model_name='yolo11s-pose.pt', output_path="lunge_output2.mp4"):
    """Process video with UPDATED thresholds, debugging and small text."""
    
    # Initialize debug counter with updated thresholds
    counter = DebugLungeCounter(model_name=model_name)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
    print(f"Using UPDATED thresholds: Down<{counter.knee_down_threshold}°, Up>{counter.knee_up_threshold}°")
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print("\nProcessing with UPDATED thresholds and DEBUG mode...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        counter.frame_count += 1
        
        # Detect pose
        keypoints, confidences, results = counter.detect_pose_yolo11(frame)
        
        # Draw skeleton
        if keypoints is not None:
            annotated_frame = results.plot()
            frame = annotated_frame
        
        # Get knee angles
        left_angle, right_angle, left_conf, right_conf = counter.get_knee_angles(keypoints, confidences)
        
        # Update lunge count
        lunge_detected, working_leg = counter.update_lunge_count(left_angle, right_angle)
        
        # Highlight knees with CORRECTED indexing
        if keypoints is not None:
            if working_leg == "left" and left_angle is not None:
                left_knee = keypoints[13]  # FIXED: proper indexing
                cv2.circle(frame, (int(left_knee[0]), int(left_knee[1])), 12, (0, 255, 255), -1)
                cv2.circle(frame, (int(left_knee[0]), int(left_knee[1])), 15, (255, 255, 0), 3)
            elif working_leg == "right" and right_angle is not None:
                right_knee = keypoints[14]  # FIXED: proper indexing
                cv2.circle(frame, (int(right_knee[0]), int(right_knee[1])), 12, (0, 255, 255), -1)
                cv2.circle(frame, (int(right_knee[0]), int(right_knee[1])), 15, (255, 255, 0), 3)
        
        # SMALL TEXT OVERLAY
        y_offset = 30
        font_scale = 0.5  # SMALLER FONT
        thickness = 1     # THINNER TEXT
        
        # Main counter (slightly larger)
        cv2.putText(frame, f"LUNGES: {counter.get_count()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (202, 255, 150), 2)
        y_offset += 25
        
        # Stage
        cv2.putText(frame, f"Stage: {counter.get_stage().upper()}", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 10), 2)
        y_offset += 20
        
        # # Frame info
        # cv2.putText(frame, f"Frame: {counter.frame_count}/{total_frames}", 
        #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        # y_offset += 20
        
        # # Thresholds
        # cv2.putText(frame, f"Thresholds: <{counter.knee_down_threshold}° >{counter.knee_up_threshold}°", 
        #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # y_offset += 15
        
        # Knee angles
        # if left_angle is not None:
        #     color = (0, 255, 0) if left_angle > counter.knee_up_threshold else (0, 0, 255) if left_angle < counter.knee_down_threshold else (255, 255, 0)
        #     cv2.putText(frame, f"L-Knee: {left_angle:.1f}°", 
        #                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        #     y_offset += 20
        
        # if right_angle is not None:
        #     color = (0, 255, 0) if right_angle > counter.knee_up_threshold else (0, 0, 255) if right_angle < counter.knee_down_threshold else (255, 255, 0)
        #     cv2.putText(frame, f"R-Knee: {right_angle:.1f}°", 
        #                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        #     y_offset += 20
        
        # Working leg
        if working_leg:
            cv2.putText(frame, f"Working: {working_leg.title()}", 
                       (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            y_offset += 20
        
        # # Debug info
        # cv2.putText(frame, f"Pose: {counter.pose_detected_count}/{counter.frame_count}", 
        #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        # y_offset += 15
        
        # cv2.putText(frame, f"Knee Det: {counter.knee_detected_count}", 
        #            (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Flash effect for lunge detection
        if lunge_detected:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 255, 0), -1)
            frame = cv2.addWeighted(frame, 0.8, overlay, 0.2, 0)
            cv2.putText(frame, "LUNGE!", (width//2 - 80, height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Write frame
        out.write(frame)
        
        # Progress update
        if counter.frame_count % 200 == 0:
            progress = (counter.frame_count / total_frames) * 100
            print(f"⏳ Progress: {progress:.1f}% - Lunges: {counter.get_count()}")
            counter.print_debug_summary()
    
    # Release resources
    cap.release()
    out.release()
    
    # Final debug summary
    print(f"\nProcessing complete with UPDATED thresholds!")
    counter.print_debug_summary()
    print(f"Output: {output_path}")
    
    return output_path, counter.get_count()


# Main execution with UPDATED thresholds
if __name__ == "__main__":
    VIDEO_PATH = "data/pilates_reformer.mp4"  # UPDATE THIS PATH
    MODEL_NAME = 'yolo11s-pose.pt'
    
    if os.path.exists(VIDEO_PATH):
        print(f"Video found: {VIDEO_PATH}")
        
        output_file, total_count = process_debug_lunge_video(VIDEO_PATH, MODEL_NAME)
