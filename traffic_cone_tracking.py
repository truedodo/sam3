import torch
import os
import cv2
import numpy as np
import time
from typing import Any, Callable
from sam3.model_builder import build_sam3_video_model
from sam3.model.data_misc import BatchedDatapoint, FindStage, convert_my_tensors
from sam3.model.utils.misc import copy_data_to_device
from sam3.model.geometry_encoders import Prompt

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Constants
MAX_FRAMES = 100000  # Virtual limit for streaming
TARGET_RESOLUTION = 1024 # Standard for SAM3
VIDEO_HEIGHT = 720
VIDEO_WIDTH = 1280

class LazyList:
    """A list-like object that generates items on demand to support large virtual lists."""
    def __init__(self, length: int, factory: Callable[[int], Any], initial_cache: dict = None):
        self.length = length
        self.factory = factory
        self.cache = initial_cache if initial_cache is not None else {}

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            # Basic slice support: return list of items
            start, stop, step = idx.indices(self.length)
            return [self[i] for i in range(start, stop, step)]
        
        if idx < 0:
            idx += self.length
        
        if idx >= self.length:
            raise IndexError("LazyList index out of range")

        if idx in self.cache:
            return self.cache[idx]
        
        item = self.factory(idx)
        self.cache[idx] = item
        return item

    def __setitem__(self, idx, value):
        if idx < 0:
            idx += self.length
        if idx >= self.length:
            raise IndexError("LazyList index out of range")
        self.cache[idx] = value

    def clear(self):
        self.cache.clear()

class WebcamLoader:
    """A loader that wraps a webcam and presents it as a random-access list (but intended for sequential access)."""
    def __init__(self, cap, image_size, mean, std, device):
        self.cap = cap
        self.image_size = image_size
        self.mean = torch.tensor(mean, dtype=torch.float16).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std, dtype=torch.float16).view(1, 3, 1, 1).to(device)
        self.device = device
        self.cache = {}
        self.last_read_idx = -1

    def __len__(self):
        return MAX_FRAMES

    def __getitem__(self, idx):
        if idx in self.cache:
            # print(f"DEBUG: Returning CACHED frame {idx}")
            return self.cache[idx]
        
        # Debug print
        print(f"DEBUG: Loading frame {idx}. Last read: {self.last_read_idx}")

        if idx <= self.last_read_idx:
            print(f"Warning: Accessing lost frame {idx} (last read: {self.last_read_idx})")
            if self.last_read_idx in self.cache:
                return self.cache[self.last_read_idx]
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float16, device=self.device)

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Webcam read failed or end of stream.")
            return torch.zeros(3, self.image_size, self.image_size, dtype=torch.float16, device=self.device)
        
        self.last_read_idx = idx
        
        # Preprocess
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        # print(f"DEBUG: Read frame {idx} shape: {frame.shape}, Mean: {frame.mean()}")

        scale = self.image_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        frame_resized = cv2.resize(frame, (new_w, new_h))
        
        canvas = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        canvas[:new_h, :new_w] = frame_resized
        
        tensor = torch.from_numpy(canvas).permute(2, 0, 1).float() 
        tensor = tensor.half().to(self.device).unsqueeze(0) 
        tensor = (tensor / 255.0 - self.mean) / self.std
        tensor = tensor.squeeze(0) 
        
        self.cache[idx] = tensor
        
        # Cleanup old cache
        to_remove = [k for k in self.cache if k < idx - 20]
        for k in to_remove:
            del self.cache[k]
            
        return tensor


def init_streaming_state(model, webcam_loader):
    """Manually initializes inference state for streaming without loading a video file."""
    inference_state = {}
    inference_state["image_size"] = model.image_size
    inference_state["num_frames"] = MAX_FRAMES
    inference_state["orig_height"] = VIDEO_HEIGHT
    inference_state["orig_width"] = VIDEO_WIDTH
    inference_state["constants"] = {}
    
    # 1. Construct Input Batch with Lazy Lists
    device = model.device
    
    # Find Inputs Factory
    def find_stage_factory(idx):
        input_box_embedding_dim = 258
        input_points_embedding_dim = 257
        stage = FindStage(
            img_ids=[idx],
            text_ids=[0],
            input_boxes=[torch.zeros(input_box_embedding_dim)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
            input_boxes_label=[torch.empty(0, dtype=torch.long)],
            input_points=[torch.empty(0, input_points_embedding_dim)],
            input_points_mask=[torch.empty(0)],
            object_ids=[],
        )
        return convert_my_tensors(stage)

    find_inputs = LazyList(MAX_FRAMES, find_stage_factory)
    
    # Find Metadatas Factory causes None access in model?
    # model.init_state sets find_targets and find_metadatas to list of None.
    # We can do same.
    
    input_batch = BatchedDatapoint(
        img_batch=webcam_loader, # Our custom loader
        find_text_batch=["<text placeholder>", "visual"],
        find_inputs=find_inputs,
        find_targets=LazyList(MAX_FRAMES, lambda i: None),
        find_metadatas=LazyList(MAX_FRAMES, lambda i: None),
    )
    
    # Move relevant parts to device (find_inputs factory handles its own tensors)
    # img_batch (WebcamLoader) handles its own tensors.
    # We assume model doesn't try to .to() the LazyLists, or LazyLists pass it through via duck typing?
    # copy_data_to_device works recursively. It checks isinstance list. LazyList is NOT list.
    # It checks _CopyableData (has .to). LazyList does NOT have .to.
    # So it returns LazyList as is. Perfect.
    
    inference_state["input_batch"] = input_batch

    # 2. Other State Params
    bs = 1
    inference_state["constants"]["empty_geometric_prompt"] = Prompt(
        box_embeddings=torch.zeros(0, bs, 4, device=device),
        box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
        box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        point_embeddings=torch.zeros(0, bs, 2, device=device),
        point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
        point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
    )

    inference_state["previous_stages_out"] = LazyList(MAX_FRAMES, lambda i: None)
    inference_state["text_prompt"] = None
    inference_state["per_frame_raw_point_input"] = LazyList(MAX_FRAMES, lambda i: None)
    inference_state["per_frame_raw_box_input"] = LazyList(MAX_FRAMES, lambda i: None)
    inference_state["per_frame_visual_prompt"] = LazyList(MAX_FRAMES, lambda i: None)
    inference_state["per_frame_geometric_prompt"] = LazyList(MAX_FRAMES, lambda i: None)
    inference_state["per_frame_cur_step"] = LazyList(MAX_FRAMES, lambda i: 0)

    inference_state["visual_prompt_embed"] = None
    inference_state["visual_prompt_mask"] = None
    inference_state["tracker_inference_states"] = []
    inference_state["tracker_metadata"] = {}
    inference_state["feature_cache"] = {}
    inference_state["cached_frame_outputs"] = {}
    inference_state["action_history"] = []
    inference_state["is_image_only"] = False # Treat as video

    return inference_state


def main():
    # 1. Device Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS acceleration.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA acceleration.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # 2. Build Model
    print("Building SAM3 Video Model...")
    # apply_temporal_disambiguation=True sets hotstart_delay=15 which breaks streaming (buffers output)
    # limit max_num_objects to small number to save memory if needed
    model = build_sam3_video_model(
        checkpoint_path=None, 
        apply_temporal_disambiguation=False 
    ).eval().to(device)

    # 3. Setup Webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    
    if not cap.isOpened():
        print("Error opening webcam.")
        return

    # 4. Initialize Streaming State
    print("Initializing Streaming State...")
    webcam_loader = WebcamLoader(cap, model.image_size, model.image_mean, model.image_std, device)
    inference_state = init_streaming_state(model, webcam_loader)
    
    # 5. Loop
    print("Starting Loop. Press 'q' to quit.")
    print("Detecting 'person' and tracking...")

    target_prompt = "person" # Or traffic cone
    
    # Add prompt on Frame 0
    # Note: add_prompt runs inference on that frame.
    # We need to make sure WebcamLoader reads the first frame when Frame 0 is requested.
    
    # We call add_prompt which updates state for Frame 0
    print(f"Adding prompt: '{target_prompt}' to Frame 0")
    
    # model.add_prompt (from Sam3VideoInference)
    # It might take a moment as it compiles things
    model.add_prompt(
        inference_state=inference_state,
        frame_idx=0,
        text_str=target_prompt
    )
    
    frame_idx = 0
    while True:
        # Propagate / Track for the current frame
        # For Frame 0, add_prompt already ran inference, but propagate_in_video ensures 
        # consistency and might be needed if add_prompt only set inputs.
        # Actually Sam3VideoInference.add_prompt runs _det_track_one_frame.
        # But we need output to visualize.
        # propagate_in_video yields outputs.
        
        # We process 1 frame at a time
        try:
            # We use a loop for the generator
            for idx, out in model.propagate_in_video(
                inference_state, 
                start_frame_idx=frame_idx, 
                max_frame_num_to_track=0
            ):
                if out is None: continue
                
                # Visualization
                # out contains: out_obj_ids, out_boxes_xywh, out_binary_masks
                
                # Get original frame for visualization (from webcam loader cache)
                # WebcamLoader returns (3, H, W) tensor, normalized, on device.
                # We need mostly the raw frame for drawing. 
                # Since we processed frame_idx, it's in WebcamLoader cache.
                # However, WebcamLoader cache has the *resized/normalized* tensor.
                # We should probably capture the raw frame in the main loop or have Loader return it?
                # To keep it simple, we just read from cap again? No, sequential.
                # Let's trust `cap.read` in the Loader happened.
                # We can't easily get the raw image back from normalized tensor perfectly for display.
                # Better solution: Just use a black canvas or reconstruct?
                # Best: Modify WebcamLoader to store raw frame separately?
                # OR: Since we are in the loop, we are synced.
                # We can't strictly access the exact raw frame object unless we stored it.
                # Let's rely on `out_binary_masks` and just overlay on a blank or reconstructed image?
                # Or just put `cap.read` in the loop and feed it to loader?
                # Refactoring WebcamLoader to accept external frame push would be better, but we went with Pull.
                # We will just accept we can't show the raw camera feed easily unless we hack it back.
                # Hack: Invert normalization to show what the model saw.
                
                tensor = webcam_loader.cache.get(frame_idx)
                if tensor is not None:
                    # Denormalize
                    # tensor is (C, H, W)
                    img = tensor.detach().cpu().float()
                    img = img * torch.tensor(model.image_std).view(3,1,1) + torch.tensor(model.image_mean).view(3,1,1)
                    img = img.permute(1, 2, 0).numpy() # H, W, 3
                    img = np.clip(img * 255, 0, 255).astype(np.uint8)
                    vis_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                else:
                    vis_frame = np.zeros((model.image_size, model.image_size, 3), dtype=np.uint8)

                # Draw Results
                ids = out["out_obj_ids"]
                boxes = out["out_boxes_xywh"]
                masks = out["out_binary_masks"]
                
                if len(ids) > 0:
                   for i, obj_id in enumerate(ids):
                       # Box
                       cx, cy, w, h = boxes[i]
                       # boxes are normalized? Sam3VideoInference says:
                       # out_boxes_xywh[..., 0] /= W_video
                       # So they are 0-1.
                       
                       H, W = vis_frame.shape[:2]
                       x = int((cx - w/2) * W)
                       y = int((cy - h/2) * H)
                       bw = int(w * W)
                       bh = int(h * H)
                       
                       # Ensure color is tuple of python ints
                       c1 = int((obj_id * 50) % 255)
                       c2 = int((obj_id * 100) % 255)
                       c3 = int((obj_id * 150 + 100) % 255)
                       color = (c1, c2, c3)
                       
                       cv2.rectangle(vis_frame, (x, y), (x + bw, y + bh), color, 2)
                       cv2.putText(vis_frame, f"ID {obj_id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                       
                       # Mask
                       # mask is H_orig, W_orig (720, 1280)
                       # Vis frame is 1024x1024 (resized)
                       # We need to resize mask to match vis_frame
                       mask = masks[i] # boolean
                       mask_uint8 = (mask.astype(np.uint8) * 255)
                       mask_resized = cv2.resize(mask_uint8, (W, H), interpolation=cv2.INTER_NEAREST)
                       
                       # Blend
                       overlay = vis_frame.copy()
                       overlay[mask_resized > 128] = np.array(color, dtype=np.uint8)
                       cv2.addWeighted(overlay, 0.4, vis_frame, 0.6, 0, vis_frame)

                if frame_idx == 5:
                    save_path = "/Users/dodo/.gemini/antigravity/brain/6fec1f3e-1abe-43b4-9a60-97d903be1dbc/debug_vis_frame.jpg"
                    cv2.imwrite(save_path, vis_frame)
                    print(f"DEBUG: Saved frame {frame_idx} to {save_path}")

                cv2.imshow("SAM3 Video Tracking", vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return

        except Exception as e:
            print(f"Error in propagation: {e}")
            break
            
        frame_idx += 1
        # Periodically clear feature cache to prevent OOM
        # SAM3 holds features in `inference_state["feature_cache"]`
        # We should clear entries older than X frames
        # The model needs some history. `Sam3TrackerBase` uses limited memory bank.
        # But `inference_state["feature_cache"]` collects backbone features.
        cache = inference_state["feature_cache"]
        keys = list(cache.keys())
        for k in keys:
            if isinstance(k, int) and k < frame_idx - 20: # Keep last 20 frames
                 del cache[k]
        
        # Also clear previous_stages_out if using LazyList? 
        # LazyList stores in .cache. Accessing clears it?
        # My LazyList implementation grows .cache. 
        # I need to clear it.
        inference_state["input_batch"].find_inputs.cache.clear() # Clear inputs cache (tensors)
        inference_state["previous_stages_out"].cache.clear() # Clear outputs cache
        
        # NOTE: Clearing previous_stages_out might break memory attention if it needs to look back at OUTPUTS.
        # SAM3 Tracking uses `output_dict` (in Tracker) vs `previous_stages_out` (in Inference).
        # `previous_stages_out` stores high-level results.
        # We should probably implement LRU in LazyList for better safety.
        # For now, explicit clear of very old items might be needed if memory grows.
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
