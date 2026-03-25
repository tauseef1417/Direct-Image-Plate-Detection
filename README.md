# Direct-Image-Plate-Detection

**Automated Detection and Morphological Analysis of Stellar Sources in Digitized Direct-Image Photographic Plates**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)](https://opencv.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Overview

This repository contains the image-processing pipeline developed for the paper:

> **Automated Detection and Morphological Analysis of Stellar Sources in Digitized Direct-Image Photographic Plates**  
> T. A. Zafar, R. Hudec  
> *Multi-Frequency Behaviour of High Energy Cosmic Sources XV (MULTIF2025)*  
> Palermo, Italy, June 2025

The pipeline detects stellar sources in 600-dpi TIFF scans of direct-image astronomical photographic plates from the **Sonneberg Observatory** archive using classical computer vision (OpenCV).

---

## How it works

```
Load TIFF (grayscale)
        ↓
Pixel-wise Inversion   [cv2.bitwise_not]
        ↓
Binary Threshold       [T = 120]
        ↓
Contour Extraction     [cv2.findContours]
        ↓
Bounding-box Filter    [w × h ≥ 15 px²]
        ↓
Morphology Descriptors [area, aspect ratio, circularity, solidity, extent]
        ↓
Detection Overlay + Statistics
```

---

## Results on Plate \#602940 (Sonneberg Observatory)

| Quantity | Value |
|---|---|
| Raw detections | 14,100 |
| Retained after filter | 8,773 (62.2%) |
| Mean area | 363.4 px |
| Median area | 120.0 px |
| Min / Max area | 30.0 / 4,896.0 px |
| Mean aspect ratio | 1.03 |
| Median aspect ratio | 1.00 |

> Near-unity aspect ratios confirm that the detected population is
> overwhelmingly composed of compact, isotropic stellar images —
> consistent with direct-image plates.

---

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

### Google Colab (recommended)

1. Open `detection_pipeline.py` and paste it as a cell in your Colab notebook.
2. Set your plate path:
```python
PLATE_PATH = '/content/drive/MyDrive/602940.tif'
SAVE_DIR   = '/content/drive/MyDrive/'
```
3. Run the cell. Output: detection overlay image + printed statistics.

### Local

```bash
python detection_pipeline.py
```
Set `PLATE_PATH` and `SAVE_DIR` at the top of the script.

---

## Morphology Descriptors

| Descriptor | Formula | Notes |
|---|---|---|
| Area | `cv2.contourArea` | Pixels inside contour |
| Aspect ratio | `w / h` | 1.0 = circle / square |
| Circularity | `4π·A / P²` | 1.0 = perfect circle |
| Solidity | `A / A_convex_hull` | 1.0 = fully convex |
| Extent | `A / A_bbox` | 1.0 = fills bounding box |

---

## Known False-Positive Classes

Three categories of non-stellar detections pass the current area filter:

| Class | Cause | Mitigation |
|---|---|---|
| Bright-star haloes | Merged diffraction rings → large connected regions | Upper area cut + circularity filter `C > 0.50` |
| Plate border artefacts | Dark frame, scratches, scanner vignetting | 50–100 px border exclusion mask |
| Annotation text | Plate inscription characters | Exclude known annotation region |

---

## Data

Plate scans are from the **Sonneberg Observatory** photographic archive,
digitized at 600 dpi using a commercial flatbed scanner.
Plate files are **not included** in this repository due to their large size (~500 MB per TIFF).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ZafarHudec2025,
  author    = {Zafar, Tauseef Ahmad and Hudec, Ren\'{e}},
  title     = {Automated Detection and Morphological Analysis of Stellar
               Sources in Digitized Direct-Image Photographic Plates},
  booktitle = {Proceedings of Science -- MULTIF2025},
  year      = {2025},
  address   = {Palermo, Italy}
}
```

---

## Related Work

> T. A. Zafar and R. Hudec,
> *Feasibility study for application of digitized Henize Mt Wilson Michigan
> Southern Sky Hα survey low-dispersive spectral plates for searches for
> anomalous and highly redshifted objects.*
> Astronomische Nachrichten, accepted 2026.

---

## Authors

- **Tauseef Ahmad Zafar** — Czech Technical University in Prague  
  tauseef1417@gmail.com
- **Prof. René Hudec** — Czech Technical University in Prague /  
  Astronomical Institute of the Czech Academy of Sciences  
  rene.hudec@gmail.com

---

## License

MIT — see [LICENSE](LICENSE) for details.
