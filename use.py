from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2.utils.visualizer import Visualizer
from detectron2.detectron2.data import MetadataCatalog
import cv2

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

# Visualize the results
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()