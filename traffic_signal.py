import cv2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

def initialize_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("configs/config.yaml")  
    cfg.MODEL.WEIGHTS = "output/ogmodel.pth"   
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.DEVICE = "cuda" 
    
    return DefaultPredictor(cfg)

drawing = False
start_point = (0, 0)
end_point = (0, 0)
line_position = None
line_drawn = False
predictor = initialize_predictor()
    
def draw_line(event, x, y, flags, params):
    global drawing, start_point, end_point, line_position, line_drawn

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        line_position = (start_point, end_point)  # Store the drawn line
        print(f"Line drawn from {start_point} to {end_point}")
        line_drawn = True

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", draw_line)
im = cv2.imread("test/street2.jpg")
outputs = predictor(im)

instances = outputs["instances"]
pred_boxes = instances.pred_boxes
pred_classes = instances.pred_classes
metadata = predictor.metadata
class_names = metadata.get("thing_classes", [])


def count_cars_crossing_line(pred_boxes, line_position):
    line_crossed_count = 0
    if line_position is None:
        return 0
    # Get the coordinates of the drawn line
    (x1, y1), (x2, y2) = line_position
    
    # y = mx + b
    # b = y - mx
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    
    for box, class_id in zip(pred_boxes, pred_classes):
        if class_names[class_id] in ("car", "truck"):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            c_x, c_y = (x_min + x_max / 2, y_min + y_max / 2)
            
            line_y = m * c_x + b
            # Check if the car crosses the line (simplified logic)
            if c_y > line_y:
                line_crossed_count += 1
    return line_crossed_count

temp_image = im.copy()
visualizer = Visualizer(temp_image[:, :, ::-1], metadata=metadata, scale=0.8)
vis_image = visualizer.draw_instance_predictions(instances.to("cpu"))
temp_image = vis_image.get_image()

while True:

    cv2.imshow("Image", temp_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    cars_crossed = count_cars_crossing_line(pred_boxes, line_position)
    if line_position and line_drawn:
        (x1, y1), (x2, y2) = line_position
        cv2.line(temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw green line
        print(f"Number of vehicles waiting: {cars_crossed}")
        line_drawn = False
    
cv2.destroyAllWindows()
    
