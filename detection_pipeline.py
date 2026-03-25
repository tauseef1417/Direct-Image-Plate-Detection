# ============================================================
#  detection_pipeline.py
#
#  Automated Detection and Morphological Analysis of Stellar
#  Sources in Digitized Direct-Image Photographic Plates
#
#  Authors : Tauseef Ahmad Zafar, René Hudec
#  Affil.  : Czech Technical University in Prague
#  Paper   : MULTIF2025, Palermo, Italy, June 2025
#
#  Platform: Google Colab (also runs locally with Python 3.8+)
#  Run     : paste as a cell in Colab, or: python detection_pipeline.py
# ============================================================

import gc
import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from google.colab.patches import cv2_imshow
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ============================================================
#  CONFIGURATION  — edit these before running
# ============================================================
PLATE_PATH     = '/content/drive/MyDrive/602940.tif'
SAVE_DIR       = '/content/drive/MyDrive/'
THRESHOLD      = 120    # inverse binary threshold value (0–255)
MIN_BOX_AREA   = 15     # minimum bounding-box area in px² (noise filter)
DISPLAY_SCALE  = 0.25   # downscale factor for overlay display (saves RAM)

# ============================================================
#  STEP 1  Load plate in grayscale
# ============================================================
print('Loading plate ...')
image = cv2.imread(PLATE_PATH, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(
        f'Cannot read plate at: {PLATE_PATH}\n'
        'Check the path and confirm the file is mounted.')

H, W = image.shape
print(f'  Size : {W} x {H} px')

# ============================================================
#  STEP 2  Pixel-wise inversion
#  Dark stellar images → bright foreground for thresholding
# ============================================================
inverted = cv2.bitwise_not(image)

# ============================================================
#  STEP 3  Global binary threshold
# ============================================================
_, thresh = cv2.threshold(inverted, THRESHOLD, 255, cv2.THRESH_BINARY)
del inverted
gc.collect()

# ============================================================
#  STEP 4  External contour extraction
# ============================================================
print('Finding contours ...')
contours, _ = cv2.findContours(thresh,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
print(f'  Raw candidates : {len(contours):,}')

# ============================================================
#  STEP 5  Filter small objects + compute morphology descriptors
# ============================================================
print('Computing descriptors ...')

areas, aspect_ratios, circularities, solidities, extents = [], [], [], [], []
kept_boxes = []   # (x, y, w, h) tuples — lightweight, no contour objects

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # ── bounding-box area filter ──────────────────────────────
    if w * h < MIN_BOX_AREA:
        continue

    kept_boxes.append((x, y, w, h))

    # Area
    a = float(cv2.contourArea(cnt))
    areas.append(a if a > 0 else float(w * h))

    # Aspect ratio  r = w / h
    aspect_ratios.append(w / h if h > 0 else 1.0)

    # Circularity  C = 4π·A / P²
    p    = cv2.arcLength(cnt, True)
    circ = (4.0 * np.pi * a / p ** 2) if p > 0 else 0.0
    circularities.append(min(circ, 1.0))

    # Solidity  S = A / A_convex_hull
    hull      = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    solidities.append(a / hull_area if hull_area > 0 else 1.0)

    # Extent  E = A / A_bounding_box
    extents.append(a / (w * h) if (w * h) > 0 else 0.0)

del contours
gc.collect()

areas         = np.array(areas,         dtype=np.float32)
aspect_ratios = np.array(aspect_ratios, dtype=np.float32)
circularities = np.array(circularities, dtype=np.float32)
solidities    = np.array(solidities,    dtype=np.float32)
extents       = np.array(extents,       dtype=np.float32)

# ============================================================
#  STEP 6  Print summary statistics
# ============================================================
n = len(kept_boxes)
print()
print('=' * 52)
print(f'  Total raw detections          : {len(areas) + (len(areas)==0):,}')
print(f'  Retained after filter         : {n:,}')
print()
print(f'  Mean area                     : {areas.mean():.1f} px')
print(f'  Median area                   : {float(np.median(areas)):.1f} px')
print(f'  Min area                      : {areas.min():.1f} px')
print(f'  Max area                      : {areas.max():.1f} px')
print()
print(f'  Mean aspect ratio (w/h)       : {aspect_ratios.mean():.2f}')
print(f'  Median aspect ratio           : {float(np.median(aspect_ratios)):.2f}')
print()
print(f'  Mean circularity              : {circularities.mean():.3f}')
print(f'  Mean solidity                 : {solidities.mean():.3f}')
print(f'  Mean extent                   : {extents.mean():.3f}')
print()
print(f'  % circularity > 0.85          : {100*(circularities>0.85).mean():.1f}%')
print(f'  % solidity    > 0.90          : {100*(solidities>0.90).mean():.1f}%')
print(f'  % extent      > 0.80          : {100*(extents>0.80).mean():.1f}%')
print('=' * 52)

# ============================================================
#  STEP 7  Detection overlay (downsampled to save RAM)
# ============================================================
H_sm  = int(H * DISPLAY_SCALE)
W_sm  = int(W * DISPLAY_SCALE)
small = cv2.resize(image, (W_sm, H_sm), interpolation=cv2.INTER_AREA)
overlay = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
del small
gc.collect()

for (x, y, w, h) in kept_boxes:
    xs = int(x * DISPLAY_SCALE);  ys = int(y * DISPLAY_SCALE)
    ws = max(int(w * DISPLAY_SCALE), 1)
    hs = max(int(h * DISPLAY_SCALE), 1)
    cv2.rectangle(overlay, (xs, ys), (xs+ws, ys+hs), (0, 210, 0), 1)

if IN_COLAB:
    cv2_imshow(overlay)
else:
    path = SAVE_DIR + 'detection_overlay.png'
    cv2.imwrite(path, overlay)
    print(f'\n  Overlay saved → {path}')

del overlay
gc.collect()

# ============================================================
#  STEP 8  Area histogram
# ============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.hist(areas, bins=20, color='#4C72B0', edgecolor='white',
        linewidth=0.7, alpha=0.88)
ax.axvline(areas.mean(), color='#DD8452', lw=1.8, linestyle='--',
           label=f'Mean = {areas.mean():.1f} px')
ax.axvline(float(np.median(areas)), color='#55A868', lw=1.8, linestyle=':',
           label=f'Median = {float(np.median(areas)):.1f} px')
ax.set_xlabel('Spot Area (pixels)')
ax.set_ylabel('Count')
ax.set_title('Distribution of Detected Spot Areas')
ax.legend()
ax.yaxis.grid(True, linestyle='--', alpha=0.35)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.show()

print('\nDone.')
