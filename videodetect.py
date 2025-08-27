from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.utils.visualizer import Visualizer
from detectron2.detectron2.data import MetadataCatalog
import cv2, torch, numpy as np
from collections import Counter
import time

class VideoCarDetector:
    def __init__(self):
        self.cfg = None
        self.predictor = None
        self.metadata = None
        self.roi = None
        self.roi_selected = False
        self.setup_model()
        
    def setup_model(self):
        """Initialize the Detectron2 model"""
        self.cfg = get_cfg()
        self.cfg.merge_from_file("configs/config.yaml")  
        self.cfg.MODEL.WEIGHTS = "output/ogmodel.pth" 
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
        self.cfg.MODEL.DEVICE = "cuda" 
        
        # Initialize predictor
        self.predictor = DefaultPredictor(self.cfg)
        self.metadata = MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0])
        print("Model loaded successfully!")
    
    def select_roi_from_frame(self, frame):
        """Allow user to select ROI from a paused frame"""
        print("\n" + "="*60)
        print("ROI SELECTION MODE")
        print("="*60)
        print("Video is paused. Select the area where you want to detect cars.")
        print("Click and drag to select the area.")
        print("Press SPACE or ENTER to confirm your selection")
        print("Press ESC to cancel")
        print("="*60)
        
        # Create a copy of the frame for ROI selection
        roi_frame = frame.copy()
        
        # Select ROI
        roi = cv2.selectROI("Select ROI for Car Detection (Press SPACE/ENTER to confirm)", 
                           roi_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI for Car Detection (Press SPACE/ENTER to confirm)")
        
        if roi[2] > 0 and roi[3] > 0:
            self.roi = roi
            self.roi_selected = True
            x, y, w, h = roi
            print(f"\nROI selected: x={x}, y={y}, width={w}, height={h}")
            print("ROI selection complete! Press ENTER in the video window to start detection...")
            return True
        else:
            print("\nNo ROI selected.")
            return False
    
    def detect_cars_in_roi(self, frame):
        """Detect cars in the selected ROI"""
        if not self.roi_selected or self.roi is None:
            return frame, 0
        
        x, y, w, h = self.roi
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        # Crop frame to ROI
        roi_frame = frame[y1:y2, x1:x2]
        
        # Run detection on cropped frame
        outputs = self.predictor(roi_frame)
        instances = outputs["instances"]
        
        # Adjust bounding box coordinates back to original frame coordinates
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            # Adjust coordinates to map back to original frame
            boxes[:, [0, 2]] += x1  # Adjust x coordinates
            boxes[:, [1, 3]] += y1  # Adjust y coordinates
            
            instances.pred_boxes.tensor = torch.tensor(boxes).to(instances.pred_boxes.tensor.device)
        
        # Create visualization
        v = Visualizer(frame[:, :, ::-1], self.metadata, scale=1.0)
        out = v.draw_instance_predictions(instances.to("cpu"))
        result_frame = out.get_image()[:, :, ::-1]
        result_frame = np.ascontiguousarray(result_frame, dtype=np.uint8)
        
        # Draw ROI rectangle
        cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add car count text
        car_count = len(instances)
        cv2.putText(result_frame, f"Cars in ROI: {car_count}", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result_frame, car_count
    
    def process_video(self, video_path):
        """Process video with ROI selection and car detection"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file '{video_path}'")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video loaded: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
        
        # Read first frame for ROI selection
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read first frame")
            cap.release()
            return
        
        # Allow user to select ROI
        if not self.select_roi_from_frame(first_frame):
            cap.release()
            return
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        print("\nStarting video playback with car detection...")
        print("Controls:")
        print("- SPACE: Pause/Resume")
        print("- 'r': Reselect ROI")
        print("- 'q' or ESC: Quit")
        print("- 's': Save current frame")
        
        frame_count = 0
        paused = False
        detection_active = False
        
        # Wait for user to press ENTER to start detection
        print("\nPress ENTER in the video window to start detection...")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("End of video reached")
                    break
                frame_count += 1
            
            # Process frame for car detection if active
            if detection_active and self.roi_selected:
                processed_frame, car_count = self.detect_cars_in_roi(frame)
                display_frame = processed_frame
                
                # Add frame info
                cv2.putText(display_frame, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(display_frame, "PAUSED - Press SPACE to resume", 
                               (10, frame_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                display_frame = frame.copy()
                if self.roi_selected:
                    x, y, w, h = self.roi
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "Press ENTER to start detection", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No ROI selected - Press 'r' to select", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Video Car Detection", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27: 
                break
            elif key == ord(' '):  # SPACE - pause/resume
                paused = not paused
                print(f"Video {'paused' if paused else 'resumed'}")
            elif key == ord('r'):  # 'r' - reselect ROI
                detection_active = False
                if self.select_roi_from_frame(frame):
                    print("ROI reselected. Press ENTER to start detection...")
            elif key == 13:  # ENTER - start/stop detection
                if self.roi_selected:
                    detection_active = not detection_active
                    print(f"Detection {'started' if detection_active else 'stopped'}")
                else:
                    print("Please select ROI first (press 'r')")
            elif key == ord('s'):  # 's' - save frame
                timestamp = int(time.time())
                filename = f"frame_{timestamp}.jpg"
                cv2.imwrite(filename, display_frame)
                print(f"Frame saved as {filename}")
            
            # Control playback speed
            if not paused:
                time.sleep(1.0 / fps)  # Maintain original video speed
        
        cap.release()
        cv2.destroyAllWindows()
        print("Video processing complete!")

def main():
    print("Video Car Detection with ROI Selection")
    print("=" * 50)
    
    # Initialize detector
    detector = VideoCarDetector()
    
    video_path = "test/carsvideo.mp4"
    
    print(f"Loading video: {video_path}")
    
    try:
        detector.process_video(video_path)
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()