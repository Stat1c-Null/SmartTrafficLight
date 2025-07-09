from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.utils.visualizer import Visualizer
from detectron2.detectron2.data import MetadataCatalog
import cv2, torch, numpy as np
from collections import Counter

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file("configs/config.yaml")  # Load the same config used for training
cfg.MODEL.WEIGHTS = "output/ogmodel.pth"  # Path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold for predictions
cfg.MODEL.DEVICE = "cuda"  # Use GPU 

# Initialize predictor with the trained model
predictor = DefaultPredictor(cfg)

im = cv2.imread("test/street2.jpg")

# Run inference
outputs = predictor(im)

#Count predictions
instances = outputs["instances"]
pred_count = len(instances)
print(f"Total detections: {pred_count}")
pred_classes = instances.pred_classes.cpu().numpy()
class_count = Counter(pred_classes)
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.thing_classes

car_data = {}

print("\nDetections per class:")
print("-" * 30)
for class_id, count in class_count.items():
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
    car_data[class_name] = count
    print(f"{class_name}: {count}")

if len(instances) > 0:
    scores = instances.scores.cpu().numpy()
    print(f"\nConfidence scores range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Average confidence: {scores.mean():.3f}")

print("\nDetailed breakdown:")
print("-" * 40)
for i in range(len(instances)):
    class_id = pred_classes[i]
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
    confidence = scores[i]
    print(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f})")

print("Detected data")
print(car_data)
print(outputs["instances"])
print(pred_classes)


#Select area of detection
roi_x1, roi_y1 = 30, 30  # Top-left corner
roi_x2, roi_y2 = 300, 200  # Bottom-right corner

#Crop image and run detection only on ROI
def detect_in_roi_crop(image, roi_coords):
    x1, y1, x2, y2 = roi_coords
    
    roi_image = image[y1:y2, x1:x2]
    
    outputs = predictor(roi_image)
    
    # Adjust bounding box coordinates back to original image coordinates
    instances = outputs["instances"]
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.cpu().numpy()
        # Adjust coordinates
        boxes[:, [0, 2]] += x1  # Adjust x coordinates
        boxes[:, [1, 3]] += y1  # Adjust y coordinates
        
        instances.pred_boxes.tensor = torch.tensor(boxes).to(instances.pred_boxes.tensor.device)
    
    return outputs, roi_image

#Use predefined ROI coordinates (crop method)
roi_coords = (roi_x1, roi_y1, roi_x2, roi_y2)
outputs, roi_image = detect_in_roi_crop(im, roi_coords)

#Show output
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Draw ROI rectangle on the output image
result_image = out.get_image()[:, :, ::-1]
# Ensure the image is in the correct format for OpenCV
result_image = np.ascontiguousarray(result_image, dtype=np.uint8)
cv2.rectangle(result_image, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)  # Green rectangle for ROI

cv2.imshow("Prediction with ROI", result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()