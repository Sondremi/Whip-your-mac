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
