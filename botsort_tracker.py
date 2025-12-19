from boxmot import BotSort
import numpy as np
from pathlib import Path

class BoTSORTTracker:
    def __init__(self):
        # Initialize BoT-SORT tracker
        self.tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device='cpu',
            half=False,
            track_high_thresh=0.3, # detections with confidence ≥ track_high_thresh are used in the first matching stage.
            track_low_thresh=0.05, # detections with confidence ≥ track_low_thresh (but < 0.3) are used in the second matching stage.
            new_track_thresh=0.4, # A detection must have confidence ≥ 0.4 to start a brand new track.
            match_thresh=0.7, # threshold for appearance feature matching.
            frame_rate=14
        )

    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, conf, cls]
        frame: current image frame (numpy array)
        returns: tracks with IDs
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets, frame)
        return tracks