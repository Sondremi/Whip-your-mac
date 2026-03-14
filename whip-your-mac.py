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

# Geometry

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
