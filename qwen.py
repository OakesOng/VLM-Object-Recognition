import cv2
import re
import torch
import time
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig

# ----------------------------------------------------------------------------------
# 1️⃣  Model & Processor Initialization
# ----------------------------------------------------------------------------------

# Tiny model to fit edge GPUs
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # replace with the actual tiny HF model ID

# Quantization config for bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # more accurate than fp4
    bnb_4bit_use_double_quant=True,      # double quantization to save memory
    bnb_4bit_compute_dtype=torch.float16 # compute in fp16 for speed
)

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# ----------------------------------------------------------------------------------
# 2️⃣  Helper Functions
# ----------------------------------------------------------------------------------

def parse_ref_boxes(text: str):
    """
    Extract clean (label, x1, y1, x2, y2) from <ref>…</ref><box>(x1,y1),(x2,y2)</box>.
    Ignores unrelated text.
    """
    # Remove markdown/code fences and role tags
    cleaned = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"^(system|user|assistant)\s*$", "", cleaned, flags=re.MULTILINE | re.IGNORECASE)

    # Strict pattern: only match valid label and integer coords
    pattern = re.compile(
        r"<ref>\s*([^<]+?)\s*</ref>\s*<box>\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*</box>",
        flags=re.DOTALL
    )

    items = []
    for m in pattern.finditer(cleaned):
        label = m.group(1).strip()
        coords = list(map(int, m.groups()[1:]))
        items.append((label, *coords))

    return items


def draw_ref_boxes(frame, ref_boxes):
    """Draw labelled bounding boxes on frame."""
    for label, x1, y1, x2, y2 in ref_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, max(1, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# ----------------------------------------------------------------------------------
# 3️⃣  Mouse coordinate tracker
# ----------------------------------------------------------------------------------

mouse_coords = (0, 0)
def mouse_move(event, x, y, flags, param):
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)

# ----------------------------------------------------------------------------------
# 3️⃣  OpenCV Camera Setup
# ----------------------------------------------------------------------------------

VIDEO_DEVICE = 0  # or "/dev/video4" for Insta360 stitched stream
cap = cv2.VideoCapture(VIDEO_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)   # lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print(f"❗Could not open video device {VIDEO_DEVICE}")
    exit(1)

print("Press 'q' to quit")

cv2.namedWindow("Edge Qwen2.5 2B Detection")
cv2.setMouseCallback("Edge Qwen2.5 2B Detection", mouse_move)

# ----------------------------------------------------------------------------------
# 4️⃣  Main Loop
# ----------------------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    h, w = frame.shape[:2]

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        # "Detect cup and output in format:\n"
                        # "<ref>[object_label]</ref><box>(x1,y1),(x2,y2)</box>\n"
                        # "Only output tags—no JSON, no explanation."
                        "Locate all cups in this image. "
                        "For each cup, output: <ref>cup</ref><box>(x1,y1),(x2,y2)</box>. "
                        "Do not output other objects."

                    )
                }
            ]
        }
    ]
    chat = processor.apply_chat_template(messages, add_generation_prompt=True)

    start = time.time()
    try:
        inputs = processor(text=[chat], images=[pil], return_tensors="pt")
        inputs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in inputs.items()}

        outputs = model.generate(**inputs, max_new_tokens=80)
        resp = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        ref_boxes = parse_ref_boxes(resp)
        if ref_boxes:
            print("→ Parsed boxes:", ref_boxes)
        else:
            print("⚠️ No <ref><box> tags found:", resp.strip())

    except Exception as e:
        print("Error during inference:", e)
        ref_boxes = []

    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.putText(frame, f"Mouse: {mouse_coords}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    draw_ref_boxes(frame, ref_boxes)
    cv2.imshow("Edge Qwen2.5 2B Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
