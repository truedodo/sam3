import torch
import os
import cv2
import numpy as np
import threading
import queue
import time
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torchvision.transforms import v2
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable MPS fallback
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# --- 1. Multi-Object Tracker Class ---
class ConeTracker:
    def __init__(self, patience=10):
        self.trackers = []  # List of cv2.TrackerKCF
        self.boxes = []     # List of [x, y, w, h]
        self.ids = []       # Unique IDs
        self.colors = []    # Color for each ID
        self.patience = patience
        self.lost_counters = []
        self.next_id = 1

    def _create_tracker(self, frame, box):
        tracker = cv2.TrackerKCF_create()
        tracker.init(frame, tuple(box))
        return tracker

    def update(self, frame):
        """Called every frame to interpolate positions."""
        active_indices = []
        
        for i, tracker in enumerate(self.trackers):
            success, box = tracker.update(frame)
            if success:
                self.boxes[i] = list(map(int, box))
                # If tracking works, we keep the object
                active_indices.append(i)
            else:
                self.lost_counters[i] += 1
                # If lost but within patience, keep it (but don't update box)
                if self.lost_counters[i] < self.patience:
                    active_indices.append(i)

        # Filter out dead trackers
        self.trackers = [self.trackers[i] for i in active_indices]
        self.boxes = [self.boxes[i] for i in active_indices]
        self.ids = [self.ids[i] for i in active_indices]
        self.colors = [self.colors[i] for i in active_indices]
        self.lost_counters = [self.lost_counters[i] for i in active_indices]

        return list(zip(self.ids, self.boxes, self.colors))

    def sync(self, frame, sam_boxes):
        """
        Called when SAM3 returns new boxes.
        Matches SAM3 boxes (x1, y1, x2, y2) to existing Trackers (x, y, w, h).
        """
        # Convert SAM3 [x1, y1, x2, y2] -> [x, y, w, h]
        new_detections = []
        for b in sam_boxes:
            x, y, x2, y2 = map(int, b)
            new_detections.append([x, y, x2-x, y2-y])

        if not self.trackers:
            # Initialize all
            for box in new_detections:
                self._add_object(frame, box)
            return

        if not new_detections:
            # Mark all current as lost
            self.lost_counters = [c + 1 for c in self.lost_counters]
            return

        # Cost Matrix (1 - IoU)
        iou_matrix = np.zeros((len(self.boxes), len(new_detections)))
        for t, trk_box in enumerate(self.boxes):
            for d, det_box in enumerate(new_detections):
                iou_matrix[t, d] = 1 - self._calculate_iou(trk_box, det_box)

        # Hungarian Algorithm
        row_ind, col_ind = linear_sum_assignment(iou_matrix)
        
        assigned_dets = set()
        
        for r, c in zip(row_ind, col_ind):
            # If overlap is good (IoU > 0.3 => Cost < 0.7)
            if iou_matrix[r, c] < 0.7:
                # Update existing tracker with FRESH SAM3 coordinates
                self.trackers[r] = self._create_tracker(frame, new_detections[c])
                self.boxes[r] = new_detections[c]
                self.lost_counters[r] = 0
                assigned_dets.add(c)
        
        # Add new objects
        for i in range(len(new_detections)):
            if i not in assigned_dets:
                self._add_object(frame, new_detections[i])

    def _add_object(self, frame, box):
        self.trackers.append(self._create_tracker(frame, box))
        self.boxes.append(box)
        self.ids.append(self.next_id)
        self.colors.append(np.random.randint(0, 255, 3).tolist())
        self.lost_counters.append(0)
        self.next_id += 1

    def _calculate_iou(self, boxA, boxB):
        # standard intersection over union
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)


# --- 2. Async SAM3 Inference Class ---
class AsyncSAM3:
    def __init__(self, resolution=480, use_half=False):
        self.device = torch.device("mps")
        self.resolution = resolution
        self.use_half = use_half
        self.frame_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        self.stopped = False
        
        # Initialize Model (Heavy lifting done once)
        print("Initializing SAM3 Model...")
        self.model = build_sam3_image_model().to(self.device)
        if self.use_half:
            self.model = self.model.half()
        
        self.processor = Sam3Processor(self.model, resolution=resolution, device=self.device)
        
        if self.use_half:
            self.processor.transform = v2.Compose([
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float16, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
            
        # Pre-compute text features
        self.text_prompt = "a person"
        print(f"Pre-computing features for: {self.text_prompt}")
        with torch.inference_mode():
            self.text_features = self.model.backbone.forward_text([self.text_prompt], device=self.device)
            if self.use_half:
                self.text_features = {k: v.half() if isinstance(v, torch.Tensor) else v for k, v in self.text_features.items()}

    def start(self):
        t = threading.Thread(target=self._inference_loop, daemon=True)
        t.start()
        return self

    def update_frame(self, frame):
        if self.frame_queue.full():
            try: self.frame_queue.get_nowait()
            except queue.Empty: pass
        self.frame_queue.put(frame)

    def get_results(self):
        if not self.result_queue.empty():
            return self.result_queue.get()
        return None

    def _inference_loop(self):
        while not self.stopped:
            if self.frame_queue.empty():
                time.sleep(0.01)
                continue
            
            frame = self.frame_queue.get()
            
            # --- Preprocessing (From your original code) ---
            color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_frame)
            w, h = pil_image.size
            max_dim = max(w, h)
            padded_image = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
            padded_image.paste(pil_image, (0, 0))

            # --- Inference ---
            with torch.inference_mode():
                inference_state = self.processor.set_image(padded_image)
                inference_state["backbone_out"].update(self.text_features)
                
                if "geometric_prompt" not in inference_state:
                    inference_state["geometric_prompt"] = self.model._get_dummy_prompt()

                output = self.processor._forward_grounding(inference_state)
                
                ops_boxes = output.get("boxes")
                ops_scores = output.get("scores")
                # Note: We are ignoring masks for the tracker to keep it fast
                # You can add mask processing back if needed, but it's heavy to pass around

                result = []
                if ops_boxes is not None and len(ops_boxes) > 0:
                    cpu_boxes = ops_boxes.cpu().numpy()
                    cpu_scores = ops_scores.cpu().numpy()
                    
                    for box, score in zip(cpu_boxes, cpu_scores):
                        if score > 0.45: # Threshold
                            result.append(box) # [x1, y1, x2, y2]
                
                # Push to main thread
                if self.result_queue.full():
                     try: self.result_queue.get_nowait()
                     except: pass
                self.result_queue.put(result)


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # Setup
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Initialize Async SAM3
    sam_thread = AsyncSAM3(resolution=480, use_half=False).start()
    
    # Initialize Tracker
    tracker_manager = ConeTracker(patience=10)

    print("System Running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # 1. Send frame to SAM3 (Doesn't block)
        sam_thread.update_frame(frame.copy())

        # 2. Check if SAM3 has new results
        sam_results = sam_thread.get_results()
        if sam_results is not None:
            # SAM3 finished! Sync trackers.
            # sam_results is list of [x1, y1, x2, y2]
            tracker_manager.sync(frame, sam_results)
            # visual indicator
            cv2.circle(frame, (30, 30), 10, (0, 255, 0), -1) 
        else:
            # visual indicator (interpolating)
            cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1)

        # 3. Update Trackers (Runs every frame, very fast)
        tracked_objects = tracker_manager.update(frame)

        # 4. Draw Results
        for (obj_id, box, color) in tracked_objects:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {obj_id}", (x, y-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Real-Time SAM3 + KCF Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            sam_thread.stopped = True
            break

    cap.release()
    cv2.destroyAllWindows()