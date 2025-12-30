# Models

This repository keeps the `models/` directory **but does not commit the actual model weights** to keep the repo size small.

## OpenCV DNN face detector (Caffe)
The face-detector backend uses:
- `models/deploy.prototxt`
- `models/res10_300x300_ssd_iter_140000.caffemodel`

These files are downloaded automatically by [`framework.models.ModelManager`](framework/models.py:14) when missing (unless downloads are disabled).

## DeepFace weights
When using the DeepFace backend, weights are downloaded on first use and stored under `models/deepface/` (the code sets `DEEPFACE_HOME` to keep downloads repo-local).

If you want to pre-download weights, just run the DeepFace pipeline once; the files will appear under `models/deepface/` and remain ignored by git.
