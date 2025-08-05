import cv2
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
# 2️⃣  Mouse coordinate tracker
# ----------------------------------------------------------------------------------

mouse_coords = (0, 0)
def mouse_move(event, x, y, flags, param):
    global mouse_coords
    if event == cv2.EVENT_MOUSEMOVE:
        mouse_coords = (x, y)

# ----------------------------------------------------------------------------------
# 3️⃣  OpenCV Camera Setup
# ----------------------------------------------------------------------------------

VIDEO_DEVICE = "/dev/video4"  # or "/dev/video4" for Insta360 stitched stream
cap = cv2.VideoCapture(VIDEO_DEVICE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2880)   # lower resolution for speed
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"❗Could not open video device {VIDEO_DEVICE}")
    exit(1)

print("Press 'q' to quit")

cv2.namedWindow("Edge Qwen2.5 2B Scene Description")
cv2.setMouseCallback("Edge Qwen2.5 2B Scene Description", mouse_move)

# ----------------------------------------------------------------------------------
# 4️⃣  Main Loop
# ----------------------------------------------------------------------------------

frame_count = 0
process_every = 10  # process 1 out of every 10 frames

while True:
    ret, frame = cap.read()
    if not ret or frame is None or frame.size == 0:
        continue

    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
    frame_count += 1
    if frame_count % process_every != 0:
        cv2.imshow("Edge Qwen2.5 2B Scene Description", frame)
        continue

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {
                    "type": "text",
                    "text": (
                        "Describe this scene with 20 words at max. "
                        "Include people, objects, surroundings, and activities. "
                    
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

        outputs = model.generate(**inputs, max_new_tokens=60)
        resp = processor.batch_decode(outputs, skip_special_tokens=True)[0]

        print("Scene description:", resp.strip())

    except Exception as e:
        print("Error during inference:", e)

    fps = 1.0 / (time.time() - start)
    cv2.putText(frame, f"{fps:.1f} FPS", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

    cv2.putText(frame, f"Mouse: {mouse_coords}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Edge Qwen2.5 2B Scene Description", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
