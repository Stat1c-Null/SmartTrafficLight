import cv2
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

# ----------- Detectron2 Setup ------------
def initialize_predictor():
    cfg = get_cfg()
    cfg.merge_from_file("configs/config.yaml")
    cfg.MODEL.WEIGHTS = "output/ogmodel.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cuda"
    return DefaultPredictor(cfg)

predictor = initialize_predictor()

# ----------- Globals ------------
drawing_line = False
start_point = (0, 0)
end_point = (0, 0)
line_position = None
line_drawn = False

polygon_points = []
polygon_closed = False
hover_point = (-1, -1)
SNAP_RADIUS = 10

def draw_shapes(event, x, y, flags, param):
    global drawing_line, start_point, end_point, line_position, line_drawn
    global polygon_points, polygon_closed, hover_point

    hover_point = (x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        if drawing_line:
            return 
        if not polygon_closed:
            if len(polygon_points) >= 3 and distance((x, y), polygon_points[0]) < SNAP_RADIUS:
                polygon_closed = True
            else:
                polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_line = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_line:
            end_point = (x, y)

    elif event == cv2.EVENT_RBUTTONUP:
        drawing_line = False
        end_point = (x, y)
        line_position = (start_point, end_point)
        line_drawn = True

def distance(p1, p2):
    return np.linalg.norm(np.array(list(p1)) - np.array(list(p2)))


# Might not need this anymore 
def count_cars_crossing_line(pred_boxes, pred_classes, line_position, class_names):
    if not line_position:
        return 0
    (x1, y1), (x2, y2) = line_position
    if x2 - x1 == 0:
        return 0
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1

    count = 0
    for box, cls in zip(pred_boxes, pred_classes):
        if class_names[cls] in ("car", "truck"):
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            c_x = (x_min + x_max) / 2
            c_y = (y_min + y_max) / 2
            line_y = m * c_x + b
            if c_y > line_y:
                count += 1
    return count

def count_cars_in_polygon(pred_boxes, pred_classes, polygon_points, class_names):
    if not polygon_closed:
        return 0
    
    count = 0
    for box, cls in zip(pred_boxes, pred_classes):
        if class_names[cls] in ("car", "truck"):
            
            # Calculates the center of the bounding box
            x_min, y_min, x_max, y_max = box.cpu().numpy()
            c_x = (x_min + x_max) / 2
            c_y = (y_min + y_max) / 2
            
            # print(c_x, c_y)
            
            inside = cv2.pointPolygonTest(np.array(polygon_points), (c_x, c_y), measureDist=False)
            print(inside)
            if inside >= 0:
                count += 1
    return count

cv2.namedWindow("Detection")
cv2.setMouseCallback("Detection", draw_shapes)

image = cv2.imread("test/street2.jpg")
outputs = predictor(image)
instances = outputs["instances"]
pred_boxes = instances.pred_boxes
pred_classes = instances.pred_classes
metadata = predictor.metadata
class_names = metadata.get("thing_classes", [])

visualizer = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
vis_image = visualizer.draw_instance_predictions(instances.to("cpu")).get_image()
temp_image = vis_image[:, :, ::-1].copy()

while True:
    display = temp_image.copy()

    # Draw polygon
    for pt in polygon_points:
        cv2.circle(display, pt, 4, (0, 255, 0), -1)
    for i in range(1, len(polygon_points)):
        cv2.line(display, polygon_points[i - 1], polygon_points[i], (255, 0, 0), 2)
    if not polygon_closed and len(polygon_points) >= 3 and distance(hover_point, polygon_points[0]) < SNAP_RADIUS:
        cv2.circle(display, polygon_points[0], 8, (0, 0, 255), 2)


    # Draw line
    if line_position:
        cv2.line(display, line_position[0], line_position[1], (0, 255, 0), 2)
        
    # Draw filled polygon
    if polygon_closed:
        overlay = display.copy()
        cv2.fillPoly(overlay, [np.array(polygon_points, np.int32)], (0, 255, 255))
        alpha = 0.1
        cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
        
        polygon_count = count_cars_in_polygon(pred_boxes, pred_classes, polygon_points, class_names)
        cv2.putText(display, f"In polygon: {polygon_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)


        
    cv2.putText(display, f"Number of Total Vehicles {len(pred_boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        

    cv2.imshow("Detection", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        line_position = None
        line_drawn = False
        polygon_points = []
        polygon_closed = False

cv2.destroyAllWindows()
