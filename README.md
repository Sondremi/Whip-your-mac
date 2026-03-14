# Whip-your-mac
Need something to do while AI is working for you? Whip your Mac...

## How to use

Requires Python 3.11

```bash
pip install opencv-python mediapipe numpy
```

```bash
python3 whip-your-mac.py
```

You can also run without the camera preview window:

```bash
python3 whip-your-mac.py --no-preview
```

When a lasso gesture is detected, the program plays a random sound from the audio folder.
