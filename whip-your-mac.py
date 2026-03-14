import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
import numpy as np
import math
import time
import os
import sys
import random
import subprocess
import argparse
import threading
from collections import deque

# Supported audio file extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".aif", ".m4a", ".ogg", ".flac"}

def angle_diff(a: float, b: float) -> float:
    """Signed angular difference in [-pi, pi]."""
    return (b - a + math.pi) % (2 * math.pi) - math.pi

def compute_arc(positions: deque) -> float:
    """
    Compute accumulated angular rotation around the centroid of the given
    positions. Returns the absolute value in radians.
    """
    if len(positions) < 3:
        return 0.0

    pts    = np.array(positions, dtype=float)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = [math.atan2(p[1] - cy, p[0] - cx) for p in pts]

    total = 0.0
    for i in range(1, len(angles)):
        total += angle_diff(angles[i - 1], angles[i])

    return abs(total)

def find_audio_folder() -> str | None:
    """Return path to the 'audio' folder next to this script, or None."""
    script_dir   = os.path.dirname(os.path.abspath(__file__))
    audio_folder = os.path.join(script_dir, "audio")
    return audio_folder if os.path.isdir(audio_folder) else None

def list_audio_files(folder: str) -> list[str]:
    """Return all supported audio files in folder."""
    files = []
    for name in os.listdir(folder):
        if os.path.splitext(name)[1].lower() in AUDIO_EXTENSIONS:
            files.append(os.path.join(folder, name))
    return sorted(files)


def play_sound_async(path: str):
    """Play an audio file in the background without blocking."""
    def _play():
        if sys.platform == "darwin":
            subprocess.run(["afplay", path],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    threading.Thread(target=_play, daemon=True).start()

class LassoDetector:
    WINDOW_SEC      = 1.2   # look-back time window (seconds)
    MIN_ARC_RAD     = 4.5   # minimum rotation ~0.7 full circles
    MIN_RADIUS_NORM = 0.04  # minimum circle radius (normalised 0-1)
    COOLDOWN_SEC    = 0.8   # minimum time between triggers

    def __init__(self):
        self._positions: deque = deque()   # (x, y, timestamp)
        self._last_trigger = 0.0
        self._flash_until  = 0.0

    def update(self, x: float, y: float, frame_w: int, frame_h: int) -> bool:
        """
        Feed a new normalised hand position (0-1).
        Returns True if a lasso gesture was detected.
        """
        now = time.time()
        self._positions.append((x, y, now))

        # Trim old points outside the time window
        cutoff = now - self.WINDOW_SEC
        while self._positions and self._positions[0][2] < cutoff:
            self._positions.popleft()

        if len(self._positions) < 8:
            return False

        if now - self._last_trigger < self.COOLDOWN_SEC:
            return False

        pts = [(p[0], p[1]) for p in self._positions]
        arc = compute_arc(deque(pts))

        arr    = np.array(pts)
        cx, cy = arr[:, 0].mean(), arr[:, 1].mean()
        radii  = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
        radius = radii.mean()

        triggered = arc >= self.MIN_ARC_RAD and radius >= self.MIN_RADIUS_NORM

        if triggered:
            self._last_trigger = now
            self._flash_until  = now + 0.4
            return True

        return False

    @property
    def is_flashing(self) -> bool:
        return time.time() < self._flash_until

    def trail_points(self, w: int, h: int) -> list[tuple[int, int]]:
        """Return pixel coordinates of the motion trail."""
        return [(int(p[0] * w), int(p[1] * h)) for p in self._positions]
