import os
import random
import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# OpenCV HSV Hue ranges (0-179)
VIBGYOR_HUES = {
    "Violet": 145,
    "Indigo": 130,
    "Blue": 110,
    "Green": 60,
    "Yellow": 30,
    "Orange": 15,
    "Red": 0  # Red is technically 0-10 and 170-180
}

def apply_smart_color_change(image_rgb, mask, target_hue_name):
    """
    Applies a specific hue to the masked region while preserving white/grey 
    reflective strips (natural look) and boosting contrast.
    """
    # 1. Get Target Hue and add slight jitter so not all "Blues" look identical
    base_hue = VIBGYOR_HUES[target_hue_name]
    hue_jitter = np.random.randint(-5, 6) # +/- 5 degrees
    target_hue_val = (base_hue + hue_jitter) % 180

    # 2. Convert to HSV
    hsv_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)
    
    # 3. Define the Masked Area
    # mask input might be (H,W) bool or (H,W,1)
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_indices = mask > 0
    
    if not np.any(mask_indices):
        return image_rgb

    # 4. Smart Coloring Logic
    # We only want to strongly color pixels that already have some saturation (the plastic body).
    # We want to preserve pixels with low saturation (the white reflective tape).
    
    # Extract channels for masked region
    s_roi = s[mask_indices].astype(np.float32)
    v_roi = v[mask_indices].astype(np.float32)
    
    # Calculate a "Colorability Factor" based on Saturation.
    # If S is low (white tape), factor is 0. If S is high (cone body), factor is 1.
    # Sigmoid-like soft threshold around S=50
    # This keeps white stripes white while coloring the cone body.
    colorability = np.clip((s_roi - 20) / 50.0, 0.0, 1.0)
    
    # 5. Apply Hue
    # We set the hue to the target, but we might want to blend it for "natural" transition 
    # if we weren't doing the sigmoid trick. With the trick, we can just set it.
    # However, to be safe, let's just force the hue on the "colorable" parts.
    
    # Create a new Hue array for the ROI
    h_roi_new = np.full_like(s_roi, target_hue_val)
    
    # 6. Contrast/Vibrance Boost
    # Make the colored parts "contrasty" and "pop"
    s_roi_new = s_roi * 0.9 # Boost saturation by 20%
    v_roi_new = v_roi * 0.8 # Slight brightness boost
    
    # Clip to valid range
    s_roi_new = np.clip(s_roi_new, 0, 255)
    v_roi_new = np.clip(v_roi_new, 0, 255)
    
    # 7. Apply back to channels using the Colorability weight
    # New pixel = (NewColor * factor) + (OldColor * (1-factor))
    # Note: For Hue, linear blending is tricky, but since we are replacing "orange" 
    # with "blue", we essentially just want to switch the hue where colorability is high.
    
    h_original = h[mask_indices].astype(np.float32)
    
    # Where colorability is high, take new Hue. Where low (white tape), keep original Hue.
    h_final = (h_roi_new * colorability) + (h_original * (1 - colorability))
    s_final = (s_roi_new * colorability) + (s_roi * (1 - colorability))
    # We generally preserve V structure entirely for shading, just slight boost
    v_final = v_roi_new 

    # Assign back
    h[mask_indices] = h_final.astype(np.uint8)
    s[mask_indices] = s_final.astype(np.uint8)
    v[mask_indices] = v_final.astype(np.uint8)
    
    # 8. Merge and Convert back to RGB
    hsv_modified = cv2.merge([h, s, v])
    rgb_modified = cv2.cvtColor(hsv_modified, cv2.COLOR_HSV2RGB)
    
    # 9. Edge Blending (Feathering)
    # We blend the modified RGB with the original RGB using the mask to fix aliased edges.
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Dynamic kernel size based on image dims
    k_size = max(3, int(min(image_rgb.shape[:2]) * 0.005)) 
    if k_size % 2 == 0: k_size += 1
    
    # Create a blurred alpha mask
    mask_blurred = cv2.GaussianBlur(mask_uint8, (k_size, k_size), 0)
    alpha = mask_blurred.astype(np.float32) / 255.0
    
    # Expand dims for broadcasting
    if alpha.ndim == 2:
        alpha = alpha[:, :, np.newaxis]
        
    # Final Composite
    # result = modified * alpha + original * (1 - alpha)
    final_image = (rgb_modified * alpha + image_rgb * (1 - alpha)).astype(np.uint8)
    
    return final_image

def main():
    input_dir = "dataset/input_data"
    output_dir = "dataset/augmented"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    if not image_files:
        print(f"No images found in {input_dir}")
        return
        
    print(f"Found {len(image_files)} images. Processing all...")
    
    print("Loading SAM3 model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    
    count_aug = 0
    
    for filename in image_files:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        # If output already exists, assume this image was processed in a prior run.
        # This lets the script be paused and restarted without reprocessing.
        if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
            print(f"  [Skipped-Restart] {filename}: already processed.")
            continue

        # print(f"Processing {filename}...")
        
        try:
            with Image.open(src_path) as img_raw:
                original_format = img_raw.format
                pil_image = img_raw.convert("RGB")
            # Create a mutable numpy copy for editing
            current_image_np = np.array(pil_image)
            
            # Run Inference
            inference_state = processor.set_image(pil_image)
            output = processor.set_text_prompt(state=inference_state, prompt="traffic cone")
            
            masks = output["masks"]
            scores = output["scores"]
            
            if isinstance(masks, torch.Tensor):
                masks = masks.cpu().numpy()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().numpy()
            
            cones_found = 0
            
            # If detections exist
            if len(masks) > 0:
                # Iterate over EACH cone instance individually
                for i, mask in enumerate(masks):
                    # Filter weak detections
                    if scores[i] < 0.25:
                        continue
                        
                    if mask.ndim == 3 and mask.shape[0] == 1:
                        mask = mask.squeeze(0)
                        
                    # Choose a random color for THIS cone
                    color_name = random.choice(list(VIBGYOR_HUES.keys()))
                    
                    # Apply color to the current state of the image
                    # We update 'current_image_np' so the next cone is drawn on top of this one (if overlapping)
                    current_image_np = apply_smart_color_change(current_image_np, mask, color_name)
                    cones_found += 1
                
                if cones_found > 0:
                    print(f"  [Augmented] {filename}: Colored {cones_found} cones.")
                    tmp_path = dst_path + ".part"
                    Image.fromarray(current_image_np).save(tmp_path, format=original_format)
                    os.replace(tmp_path, dst_path)
                    count_aug += 1
                else:
                    # SAM detected something but score was low
                    print(f"  [Skipped] {filename}: Low confidence detection.")
                    tmp_path = dst_path + ".part"
                    pil_image.save(tmp_path, format=original_format)
                    os.replace(tmp_path, dst_path)
            else:
                print(f"  [Copy] {filename}: No cones found.")
                tmp_path = dst_path + ".part"
                pil_image.save(tmp_path, format=original_format)
                os.replace(tmp_path, dst_path)
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            
    print(f"Done! Augmented {count_aug} images.")

if __name__ == "__main__":
    main()