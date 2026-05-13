# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

## [0.1.0] - 2025-05-13

### Added

- OpenCV DNN-based face detection with optional CUDA acceleration.
- Optional DeepFace backend (RetinaFace, MTCNN) with alignment.
- Detection caching with configurable LRU cache.
- Multi-scale detection for improved recall on small faces.
- Adaptive luminance preprocessing for low-contrast images.
- `FaceDetectorDNN` and `FaceDetectorDeepFace` public API.
- Typed package (`py.typed` marker).
