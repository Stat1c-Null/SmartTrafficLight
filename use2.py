from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2, torch, numpy as np
from collections import Counter

def simple_roi_car_detection():
    #simple ROI selection using OpenCV's built-in selector
    
    # Load configuration
    cfg = get_cfg()
    cfg.merge_from_file("configs/config.yaml")  
    cfg.MODEL.WEIGHTS = "output/ogmodel.pth" 
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Confidence threshold
    cfg.MODEL.DEVICE = "cuda" 
    
    # Initialize predictor
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    print("Loading image")
    image = cv2.imread("test/street2.jpg")
    
    if image is None:
        print("Error: Could not load image'")
        return
    
    # instructions
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("="*60)
    print("Click and drag to select the area. Press SPACE or ENTER to confirm your selection")
    print("Press ESC to cancel and exit")
    print("="*60)
    
    roi = cv2.selectROI("Select ROI for Car Detection", image, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI for Car Detection")
    
    # Check if ROI was selected (width and height > 0)
    if roi[2] > 0 and roi[3] > 0:
        print(f"\nROI selected: x={roi[0]}, y={roi[1]}, width={roi[2]}, height={roi[3]}")
        
        # Convert ROI format from (x, y, w, h) to (x1, y1, x2, y2)
        x, y, w, h = roi
        roi_coords = (x, y, x + w, y + h)
        x1, y1, x2, y2 = roi_coords
        
        print(f"Running car detection on selected area...")
        
        # Crop image to ROI
        roi_image = image[y1:y2, x1:x2]
        
        # Run detection on cropped image
        outputs = predictor(roi_image)
        
        instances = outputs["instances"]
        
        # Adjust bounding box coordinates back to original image coordinates
        if len(instances) > 0:
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            # Adjust coordinates to map back to original image
            boxes[:, [0, 2]] += x1  # Adjust x coordinates
            boxes[:, [1, 3]] += y1  # Adjust y coordinates
            
            instances.pred_boxes.tensor = torch.tensor(boxes).to(instances.pred_boxes.tensor.device)

        print_detection_statistics(instances, metadata)
        
        print("Generating visualization...")
        v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
        out = v.draw_instance_predictions(instances.to("cpu"))
        
        result_image = out.get_image()[:, :, ::-1]
        result_image = np.ascontiguousarray(result_image, dtype=np.uint8)
        
        # Draw ROI rectangle on the result
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add text overlay with detection count
        detection_count = len(instances)
        #cv2.putText(result_image, f"Cars detected in ROI: {detection_count}", 
                   #(x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add instructions text
        #cv2.putText(result_image, "Press any key to exit", 
                   #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display result
        cv2.imshow("Car Detection Results", result_image)
        print(f"\nDetection complete! Found {detection_count} cars in the selected area.")
        print("Press any key in the result window to exit...")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("\nNo ROI selected. Exiting...")
        return

def print_detection_statistics(instances, metadata):
    pred_count = len(instances)
    print(f"\n" + "="*50)
    print("DETECTION RESULTS")
    print("="*50)
    print(f"Total detections in ROI: {pred_count}")
    
    if pred_count > 0:
        # Count predictions by class
        pred_classes = instances.pred_classes.cpu().numpy()
        class_count = Counter(pred_classes)
        class_names = metadata.thing_classes
        
        print(f"\nDetections by class:")
        print("-" * 30)
        for class_id, count in class_count.items():
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            print(f"  {class_name}: {count}")

        scores = instances.scores.cpu().numpy()
        print(f"\nConfidence Statistics:")
        print("-" * 30)
        print(f"  Minimum confidence: {scores.min():.3f}")
        print(f"  Maximum confidence: {scores.max():.3f}")
        print(f"  Average confidence: {scores.mean():.3f}")
        print(f"\nDetailed Breakdown:")
        print("-" * 40)
        for i in range(len(instances)):
            class_id = pred_classes[i]
            class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
            confidence = scores[i]
            print(f"  Detection {i+1}: {class_name} (confidence: {confidence:.3f})")
    else:
        print("\nNo cars detected in the selected area.")
    
    print("="*50)

def main():
    print("Car Detection with ROI Selection")
    print("=" * 40)
    
    try:
        simple_roi_car_detection()
    except Exception as e:
        print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main()