from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.utils.visualizer import Visualizer
from detectron2.detectron2.data import MetadataCatalog
import cv2


# Import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#Default Configs for Model
cfg = get_cfg()

cfg = get_cfg()
cfg.merge_from_file("configs/config.yaml")  # Load the same config used for training
cfg.MODEL.WEIGHTS = "output/ogmodel.pth"  # Path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold for predictions
cfg.MODEL.DEVICE = "cuda"  # Use GPU if available, otherwise "cpu"


# Set a confidence threshold for predictions.
# Only detections with a confidence score above this will be shown.

# Load pre-trained weights for the chosen model.
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # Adjust as needed (e.g., 0.7 for higher precision)



# The DefaultPredictor handles the model loading, pre-processing, and inference.
predictor = DefaultPredictor(cfg)
im = cv2.imread("test/street2.jpg")

# Run inference
outputs = predictor(im)





# Run the predictor on the input image.
# The `outputs` will contain predictions like bounding boxes, classes, and scores.
outputs = predictor(im)


# Get the metadata for the COCO dataset (used for class names)
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# Draw the instance predictions on the image.
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# Display the image with predictions
print("\nImage with Faster R-CNN Detections:")
cv2_imshow(out.get_image()[:, :, ::-1])

# --- Step 6: Access and Print Prediction Details (Optional) ---
instances = outputs["instances"].to("cpu") # Move to CPU for easier access

# Predicted bounding boxes (xyxy format: [x_min, y_min, x_max, y_max])
pred_boxes = instances.pred_boxes.tensor.numpy()

# Predicted class labels (integer IDs)
pred_classes = instances.pred_classes.numpy()

# Predicted confidence scores
scores = instances.scores.numpy()

print("\nDetected Objects:")
for i in range(len(pred_boxes)):
    box = pred_boxes[i]
    class_id = pred_classes[i]
    score = scores[i]
    class_name = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).thing_classes[class_id]
    print(f"  Object: {class_name}, Confidence: {score:.2f}, Bounding Box: [{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]")
