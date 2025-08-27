# Import the InferencePipeline object
from inference import InferencePipeline
from dotenv import load_dotenv
import cv2
import os

load_dotenv()
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MEDIA_URL = os.getenv("MEDIA_URL")


def my_sink(result, video_frame):
    if result.get("output_image"): # Display an image from the workflow response
        
        image = result["output_image"].numpy_image
        cars_in_zone = int(result["cars_in_zone"])
        cv2.putText(image, 
                    f"Cars in Zone: {cars_in_zone}",
                    (250, 100),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.3,
                    color=(255,255,255),
                    thickness=1)
        
        # push json to a time series database?
        cv2.imshow("Workflow Image", image)
        cv2.waitKey(1)


pipeline = InferencePipeline.init_with_workflow(
    api_key=ROBOFLOW_API_KEY,
    workspace_name="smart-traffic-light-muvdo",
    workflow_id="detect-count-and-visualize-2",
    video_reference=MEDIA_URL, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    max_fps=30,
    on_prediction=my_sink
)
pipeline.start() #start the pipeline
pipeline.join() #wait for the pipeline thread to finish
