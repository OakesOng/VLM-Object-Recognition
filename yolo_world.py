import cv2
import time
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld


# ------------------------------------------------------------------------------
# 1️⃣  Model Initialization
# ------------------------------------------------------------------------------

classes = ["person wearing red shirt", "wallet"]  # <-- class list you will use in labels
model = YOLOWorld(model_id="yolo_world/l")
model.set_classes(classes)

# ------------------------------------------------------------------------------
# 2️⃣  Annotators
# ------------------------------------------------------------------------------

bbox_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(
    text_thickness=1,
    text_scale=0.7,
    text_color=sv.Color.WHITE
)

# ------------------------------------------------------------------------------
# 3️⃣  OpenCV Camera Setup
# ------------------------------------------------------------------------------

VIDEO_DEVICE = "/dev/video4"
cap = cv2.VideoCapture(VIDEO_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❗Could not open capture device {VIDEO_DEVICE}")
    exit(1)

print("Press 'q' to quit")

# ------------------------------------------------------------------------------
# 4️⃣  Real-time Loop
# ------------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    start = time.time()

    # Inference
    results = model.infer(frame, confidence=0.08)
    detections = sv.Detections.from_inference(results).with_nms(threshold=0.1)

    # Generate labels using class names + confidence
    labels = [
        f"{classes[class_id]} {confidence:.3f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate frame
    annotated = bbox_annotator.annotate(frame.copy(), detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    # Display FPS
    fps = 1.0 / (time.time() - start)
    cv2.putText(annotated, f"{fps:.1f} FPS", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("YOLO-World Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
