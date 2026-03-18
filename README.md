# Whip-your-mac
Need something to do while AI is working for you? Whip your Mac...

## How to use

Requires Python 3.12+

```bash
uv sync
```

```bash
uv run whip-your-mac.py
```

You can also run without the camera preview window:

```bash
uv run whip-your-mac.py --no-preview
```

When a lasso gesture is detected, the program plays a random sound from the audio folder.
