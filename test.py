import torch
import os

# Enable MPS fallback for missing ops like _assert_async
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

from torchvision.transforms import v2

# 1. Initialize Model and Processor

device = torch.device("mps")
print(f"Using device: {device}")

# Build model
resolution = 480
use_half = False

print(f"Initializing SAM3 with resolution={resolution}, half_precision={use_half}...")

model = build_sam3_image_model().to(device)
if use_half:
    model = model.half()

processor = Sam3Processor(model, resolution=resolution, device=device)

# OVERRIDE Transform for FP16 and speed
if use_half:
    from torchvision.transforms import v2
    processor.transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(resolution, resolution)),
        v2.ToDtype(torch.float16, scale=True), # FP16 here
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

# 2. Setup Camera Feed
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

frame_count = 0
process_every_n_frames = 1 # We can try every frame now!

print("Starting camera feed... Press 'q' to quit.")

# Store results to persist them between skipped frames
last_masks, last_boxes, last_scores = None, None, None

# Pre-compute text features once
text_prompt = "a traffic cone"
print(f"Pre-computing features for prompt: '{text_prompt}'...")
with torch.inference_mode():
    text_features = model.backbone.forward_text([text_prompt], device=device)
    if use_half:
        # Cast text features to half
        text_features = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in text_features.items()}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Run Inference (Optimized)
    if frame_count % process_every_n_frames == 0:
        # Convert OpenCV BGR frame to PIL RGB
        color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_frame)

        # Pad to square to preserve aspect ratio
        w, h = pil_image.size
        max_dim = max(w, h)
        padded_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
        padded_image.paste(pil_image, (0, 0))

        with torch.inference_mode():
            # 1. Encode Image
            inference_state = processor.set_image(padded_image)
            
            # 2. Inject cached text features (avoids running text encoder every frame)
            inference_state["backbone_out"].update(text_features)
            
            # 3. Ensure geometric prompt is initialized
            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = model._get_dummy_prompt()
                if use_half:
                     # Cast dummy prompt if needed (it handles boxes lists usually, but check internal tensors)
                     # geometric_prompt is a custom class, we might need to cast its internal tensors if they are init as fp32
                     # Usually safe as it converts inputs on append.
                     pass

            # 4. Run Decoder (Grounding)
            output = processor._forward_grounding(inference_state)
            
            ops_masks = output.get("masks")
            ops_boxes = output.get("boxes")
            ops_scores = output.get("scores")

            if ops_boxes is not None and len(ops_boxes) > 0:
                last_masks = ops_masks.cpu().numpy()
                last_boxes = ops_boxes.cpu().numpy()
                last_scores = ops_scores.cpu().numpy()
            else:
                last_masks, last_boxes, last_scores = None, None, None

    # 4. Visualization
    if last_boxes is not None:
        # Draw all at once if possible? 
        # OpenCV drawing is sequential.
        for i, box in enumerate(last_boxes):
            score = float(last_scores[i])
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                
                # Draw Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Det: {score:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw Mask
                if last_masks is not None:
                    mask = last_masks[i].squeeze()
                    
                    # Crop mask to original frame size (removes padding)
                    # The mask is returned in max_dim x max_dim, we only need the top-left actual image area
                    if mask.shape[0] > frame.shape[0] or mask.shape[1] > frame.shape[1]:
                        mask = mask[:frame.shape[0], :frame.shape[1]]

                    if (mask.shape[0] != frame.shape[0]) or (mask.shape[1] != frame.shape[1]):
                        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    mask_indices = mask > 0.5
                    
                    # Apply simple tint
                    # Create a colored overlay only for the mask region to avoid full frame copy/blend
                    roi = frame[mask_indices]
                    # Blend: 0.6 * original + 0.4 * green
                    # Green is [0, 255, 0]
                    # blended = 0.6*roi + 0.4*green
                    # This is faster than whole frame addWeighted
                    
                    # Vectorized blend on ROI
                    blended = (roi * 0.6 + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
                    frame[mask_indices] = blended

    cv2.imshow('SAM 3 Live Cone Detection', frame)
    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


