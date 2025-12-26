import torch
import time
import os
import cv2
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def benchmark(resolution=336, use_half=False):
    device = torch.device("mps")
    print(f"Benchmarking with Resolution: {resolution}, Half Precision: {use_half}")
    
    # 1. Build model
    model = build_sam3_image_model().to(device)
    if use_half:
        model = model.half()
        
    processor = Sam3Processor(model, resolution=resolution, device=device)
    if use_half:
        from torchvision.transforms import v2
        processor.transform = v2.Compose([
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(resolution, resolution)),
            v2.ToDtype(torch.float16, scale=True), # FP16 here
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
    # 2. Dummy Input
    # Create a random image of typical camera size
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    pil_image = Image.fromarray(frame)
    if use_half:
         # Note: Processor likely handles dtype conversion internally or we might need to cast inputs
         pass

    # 3. Pre-compute text features
    text_prompt = "a traffic cone"
    with torch.inference_mode():
        # Text encoder usually stays in FP32 often, or check if it supports half
        # We will try half if requested
        if use_half:
            # Cast text features manually if needed, but model.backbone.forward_text should handle it if model is half
            pass
            
        print("Pre-computing text features...")
        s = time.time()
        text_features = model.backbone.forward_text([text_prompt], device=device)
        if use_half:
             # Ensure text features are half if the rest of the pipeline expects it
             text_features = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in text_features.items()}
        print(f"Text feature computation: {time.time() - s:.4f}s")

    # 4. Warmup
    print("Warming up...")
    for _ in range(5):
        with torch.inference_mode():
            inference_state = processor.set_image(pil_image)
            inference_state["backbone_out"].update(text_features)
            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = model._get_dummy_prompt()
                if use_half:
                   inference_state["geometric_prompt"] = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in inference_state["geometric_prompt"].items()}
            
            output = processor._forward_grounding(inference_state)
            torch.mps.synchronize()

    # 5. Run Benchmark
    iterations = 20
    print(f"Running {iterations} iterations...")
    times = []
    
    for _ in range(iterations):
        t0 = time.time()
        with torch.inference_mode():
            inference_state = processor.set_image(pil_image)
            inference_state["backbone_out"].update(text_features)
            if "geometric_prompt" not in inference_state:
                inference_state["geometric_prompt"] = model._get_dummy_prompt()
                if use_half:
                   inference_state["geometric_prompt"] = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in inference_state["geometric_prompt"].items()}

            output = processor._forward_grounding(inference_state)
            
            # Simulate fetching masks (sync point)
            ops_masks = output.get("masks")
            torch.mps.synchronize()
            
        t1 = time.time()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    fps = 1.0 / avg_time
    print(f"Average Inference Time: {avg_time*1000:.2f}ms")
    print(f"FPS: {fps:.2f}")
    return fps

if __name__ == "__main__":
    # Baseline
    print("--- BASELINE (1008, FP32) ---")
    try:
        benchmark(resolution=1008, use_half=False)
    except Exception as e:
        print(f"Baseline failed: {e}")

    # Optimization 1: Resolution
    print("\n--- OPTIMIZATION 1 (336, FP32) ---")
    try:
        benchmark(resolution=336, use_half=False)
    except Exception as e:
        print(f"Opt 1 failed: {e}")

    # Optimization 2: Half Precision
    print("\n--- OPTIMIZATION 2 (336, FP16) ---")
    try:
        benchmark(resolution=336, use_half=True)
    except Exception as e:
        print(f"Opt 2 failed: {e}")
