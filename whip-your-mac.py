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
