import torch
import cv2
import os
import tempfile
import shutil
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import save_masklet_video

# ================= CONFIGURATION =================
VIDEO_PATH = "/Users/dodo/Downloads/testvid.mp4"
TARGET_FPS = 1  # Set your desired input frame rate here
OUTPUT_VIDEO_PATH = "output.mp4"
TEXT_PROMPT = "a cone"
# =================================================

def resample_video(input_path, output_path, target_fps):
    """
    Creates a temporary video file with the target frame rate by skipping frames.
    """
    cap = cv2.VideoCapture(input_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    
    if target_fps > original_fps:
        print(f"Warning: Target FPS {target_fps} > Original FPS {original_fps}. Using original.")
        target_fps = original_fps

    # Calculate skip interval (stride)
    skip_interval = max(1, int(round(original_fps / target_fps)))
    print(f"Resampling video: {original_fps:.2f} FPS -> {target_fps} FPS (Stride: {skip_interval})")

    # Get dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (width, height))
    
    frame_count = 0
    written_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % skip_interval == 0:
            out.write(frame)
            written_count += 1
        
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"Resampled video saved with {written_count} frames.")
    return written_count

# Create a temporary file for the resampled video
# We use a temp directory to avoid clutter and ensure cleanup
with tempfile.TemporaryDirectory() as temp_dir:
    temp_video_path = os.path.join(temp_dir, "temp_resampled.mp4")
    
    # 1. Resample the video
    resample_video(VIDEO_PATH, temp_video_path, TARGET_FPS)

    print("Building SAM3 video predictor...")
    video_predictor = build_sam3_video_predictor(video_loader_type="lazy_cv2")

    print("Starting session with resampled video...")
    response = video_predictor.handle_request(
        request=dict(
            type="start_session",
            resource_path=temp_video_path, # Use the resampled video
            offload_video_to_cpu=True,
        )
    )
    session_id = response["session_id"]
    print(f"Session started with ID: {session_id}")

    print("Adding prompt...")
    response = video_predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=TEXT_PROMPT,
        )
    )

    print("Propagating in video...")
    inference_outputs = {}
    for result in video_predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        inference_outputs[result["frame_index"]] = result["outputs"]

    print(f"Propagation complete. Collected outputs for {len(inference_outputs)} frames.")

    print("Loading video frames for visualization...")
    video_frames = []
    # Read from the TEMP video so dimensions and frame counts match the inference results
    cap = cv2.VideoCapture(temp_video_path) 
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(video_frames)} frames.")

    print(f"Saving visualized video to {OUTPUT_VIDEO_PATH}...")
    # Try passing fps argument if save_masklet_video supports it, otherwise it defaults
    try:
        save_masklet_video(video_frames, inference_outputs, OUTPUT_VIDEO_PATH, fps=TARGET_FPS)
    except TypeError:
        # Fallback if the specific version of save_masklet_video doesn't accept 'fps'
        save_masklet_video(video_frames, inference_outputs, OUTPUT_VIDEO_PATH)
        
    print("Done!")