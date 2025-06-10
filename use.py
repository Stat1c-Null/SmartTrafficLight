from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2, torch
from collections import Counter

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file("configs/config.yaml")  # Load the same config used for training
cfg.MODEL.WEIGHTS = "output/ogmodel.pth"  # Path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold for predictions
cfg.MODEL.DEVICE = "cuda"  # Use GPU if available, otherwise "cpu"

# Initialize predictor with the trained model
predictor = DefaultPredictor(cfg)

# Load an image
im = cv2.imread("test/street2.jpg")

# Run inference
outputs = predictor(im)

#Count predictions
instances = outputs["instances"]
pred_count = len(instances)
print(f"Total detections: {pred_count}")
pred_classes = instances.pred_classes.cpu().numpy()
class_count = Counter(pred_classes)

# Get class names from metadata
metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
class_names = metadata.thing_classes

#Store counts for each class
car_data = {}

print("\nDetections per class:")
print("-" * 30)
for class_id, count in class_count.items():
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
    car_data[class_name] = count
    print(f"{class_name}: {count}")

#Print confidence scores for each detection
if len(instances) > 0:
    scores = instances.scores.cpu().numpy()
    print(f"\nConfidence scores range: {scores.min():.3f} - {scores.max():.3f}")
    print(f"Average confidence: {scores.mean():.3f}")

#Detailed breakdown with confidence scores
print("\nDetailed breakdown:")
print("-" * 40)
for i in range(len(instances)):
    class_id = pred_classes[i]
    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
    confidence = scores[i]
    print(f"Detection {i+1}: {class_name} (confidence: {confidence:.3f})")

print("Detected data")
print(car_data)

# Visualize the results
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()