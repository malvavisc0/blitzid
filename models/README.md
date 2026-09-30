# Models

This repository keeps the `models/` directory **but does not commit the actual model weights** to keep the repo size small.

## SCRFD-2.5G face detector (ONNX)
The face-detector backend uses:
- `scrfd_2.5g.onnx` — InsightFace `buffalo_m` detection weights (SCRFD-2.5G)

This file is downloaded automatically by [`ModelManager`](../src/blitzid/_models.py) when missing (unless downloads are disabled) and stored under the platformdirs user cache (`user_cache_dir("blitzid")/models/`). Pass `model_dir=Path("models")` to `FaceDetectorDNN` to keep weights here instead.
