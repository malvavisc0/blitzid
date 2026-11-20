# Third-party notices

blitzid's source code is MIT-licensed; see [LICENSE](LICENSE). Model
weights are downloaded at runtime or baked into Docker images and carry
their own licenses, separate from blitzid's code license.

## Model weights

| Component | Use | License | Upstream |
|---|---|---|---|
| YuNet `face_detection_yunet_2026may.onnx` | face detection | MIT | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) |
| SFace (`sface_2021dec.onnx`, upstream `face_recognition_sface_2021dec.onnx`) | face recognition | Apache-2.0 | [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface) |
| FairFace `fairface.onnx` | age / gender / race attributes | CC BY 4.0 | [FairFace](https://github.com/joojs/fairface) via [fairface-onnx](https://github.com/yakhyo/fairface-onnx) |
| MiniFASNetV2 (`minifasnet_v2.onnx`, upstream `2.7_80x80_MiniFASNetV2.onnx`) | passive anti-spoofing | Apache-2.0 | [MiniFASNet](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) via [ONNX export](https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx) |
| PP-OCR models (detection, classification, recognition) | text reading | Apache-2.0 | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) via [RapidOCR](https://github.com/RapidAI/RapidOCR) |

YuNet notice: Copyright (c) 2020 Shiqi Yu. The MIT text is reproduced in
[LICENSES/YuNet-MIT.txt](LICENSES/YuNet-MIT.txt).

Apache-2.0 components are licensed under the Apache License, Version
2.0; the full text is in [LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt).

FairFace attribution: Karkkainen, K. and Joo, J., "FairFace: Face
Attribute Dataset for Balanced Race, Gender, and Age for Bias
Measurement," WACV 2021. The model weights are distributed under CC BY
4.0; preserve this attribution when redistributing the weights.

## Runtime libraries

- [OpenCV](https://opencv.org/) — Apache-2.0
- [ONNX Runtime](https://onnxruntime.ai) — MIT
- [NumPy](https://numpy.org) — BSD-3-Clause
- [zxing-cpp](https://github.com/zxing-cpp/zxing-cpp) — Apache-2.0
- [RapidOCR](https://github.com/RapidAI/RapidOCR) — Apache-2.0

Python dependencies installed from PyPI carry their own metadata and
licenses; `pip`/`uv` package metadata is the source of truth for those.
