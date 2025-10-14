import supervision as sv
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
import cv2 as cv
import os
import numpy as np
from typing import Generator
from dataclasses import dataclass


#priority queue for traffic lights
priority_queue = []

#class made to represent traffic light positions (there should be 4 instances)
class traffic_count:

    def _init_(self, object_count, isred, isyellow, isgreen):
      self.object_count = int(object_count)
      self.isred = bool(isred)
      self.isyellow = bool(isyellow)
      self.isgreen = bool(isgreen)

# ↓ helper function to read frames from source video

def generate_frames(video_file: str) -> Generator[np.ndarray, None, None]:
    video = cv.VideoCapture(video_file)

    while video.isOpened():
        success, frame = video.read()

        if not success:
            break

        yield frame

    video.release()


# ↓ stores information about output video file, width and height of the frame must be equal to input video
@dataclass(frozen=True)
class VideoConfig:
    fps: float
    width: int
    height: int

# ↓ reate cv2.VideoWriter object that we can use to save output video

def get_video_writer(target_video_path: str, video_config: VideoConfig) -> cv.VideoWriter:
    video_target_dir = os.path.dirname(os.path.abspath(target_video_path))
    os.makedirs(video_target_dir, exist_ok=True)
    return cv.VideoWriter(
        target_video_path,
        fourcc=cv.VideoWriter_fourcc(*"mp4v"),
        fps=video_config.fps,
        frameSize=(video_config.width, video_config.height),
        isColor=True
    )

# Load the trained model configuration
cfg = get_cfg()
cfg.merge_from_file("configs/config.yaml")  # Load the same config used for training
cfg.MODEL.WEIGHTS = "ogmodel.pth"  # Path to the trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set confidence threshold for predictions
cfg.MODEL.DEVICE = "cpu"  # Use GPU if available, otherwise "cpu"

# Initialize predictor with the trained model
predictor = DefaultPredictor(cfg)

#configure the video
vid = cv.VideoCapture('testvideos/testvideo1.mp4')
video_config = VideoConfig(
    fps=vid.get(cv.CAP_PROP_FPS),
width = int(vid.get(cv.CAP_PROP_FRAME_WIDTH)),
height = int(vid.get(cv.CAP_PROP_FRAME_HEIGHT)))
vid.release()
video_writer = get_video_writer(
    target_video_path= 'detectron2_repo',
    video_config=video_config)

#for each frame count the number of detections
frame_iterator = iter(generate_frames(video_file= 'testvideos/testvideo1.mp4'))
detections_per_frame = []

for frame in frame_iterator:

    output = predictor(frame)

    detections = sv.Detections.from_detectron2(output)
    print(len(detections))

    detections_per_frame.append(len(detections))

    #check if detections per frame is decreasing through gradient
    if len(detections_per_frame) > 1 and np.gradient(detections_per_frame)[-1] < 0:
        print("turn yellow")

    #if detections 0 always be red
    if len(detections) == 0 : print("turn red")

    # Visualize the results
    v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    cv.imshow("Prediction", out.get_image()[:, :, ::-1])
    cv.waitKey(500)
    cv.destroyAllWindows()



video_writer.release()

print(detections_per_frame)
