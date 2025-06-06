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

camera = cv2.VideoCapture(0)

while True:
  try:
    _, img = camera.read()

    outputs = predictor(img)

    v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("Prediction", out.get_image()[:, :, ::-1])
  except Exception as e:
    print("Exception happened: " + e)

  #Press escape to close
  k = cv2.waitKey(30) & 0xff
  if k == 27:
    break
  camera.release
