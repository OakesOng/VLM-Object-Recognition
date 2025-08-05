import cv2
import time
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld

# Initialize model — choose 'l' if you need more accuracy; 's' if you want speed (~74 FPS on V100)
model = YOLOWorld(model_id="yolo_world/l")
# Set to detect only humans
model.set_classes(["fire extinguisher"])

# Annotators for bounding boxes and labels
bbox_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.7, text_color=sv.Color.WHITE)

# OpenCV stream configuration
VIDEO_DEVICE = "/dev/video4"
cap = cv2.VideoCapture(VIDEO_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❗Could not open capture device {VIDEO_DEVICE}")
    exit(1)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    start = time.time()
    # Run inference (uses GPU if available)
    results = model.infer(frame, confidence=0.1)
    detections = sv.Detections.from_inference(results)

    annotated = bbox_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections)

    fps = 1.0 / (time.time() - start)
    cv2.putText(annotated, f"{fps:.1f} FPS", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("YOLO-World Person Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
