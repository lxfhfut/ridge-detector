"""
Visual verification tests for ridge detector fixes.
Saves results to data/results_new/ for side-by-side comparison with data/results/.

Each test targets specific fixes and image characteristics.
"""
import os
import numpy as np
import imageio.v3 as iio
from ridge_detector import RidgeDetector

OUT_DIR = "data/results_new"
IMG_DIR = "data/images"
os.makedirs(OUT_DIR, exist_ok=True)


def run(name, image_path, **kwargs):
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"Image: {image_path}")
    print(f"Params: {kwargs}")
    det = RidgeDetector(**kwargs)
    det.detect_lines(image_path)
    prefix = name.replace(" ", "_")
    det.save_results(OUT_DIR, prefix=prefix, draw_junc=True)
    print(f"  Contours: {len(det.contours)}, Junctions: {len(det.junctions)}")
    for i, c in enumerate(det.contours[:5]):
        print(f"    Contour {i}: {c.num} pts, class={c.cont_class}")
    print(f"  Results saved to {OUT_DIR}/{prefix}_*")
    return det


# ---- img0: road (light line on dark background) ----
# Exercises fix #5 (scale selection for light lines)
run("img0_light_line",
    f"{IMG_DIR}/img0.jpg",
    line_widths=np.arange(3, 7),
    low_contrast=50, high_contrast=100,
    min_len=15, dark_line=False, estimate_width=True,
    extend_line=True)

# ---- img1: neurons (dark lines, many junctions) ----
# Exercises fix #1 (junction fallback), #2 (normx sign), #3 (Taylor offset), #4 (junction remap)
run("img1_dark_junctions",
    f"{IMG_DIR}/img1.jpg",
    line_widths=np.arange(1, 5),
    low_contrast=50, high_contrast=100,
    min_len=10, dark_line=True, estimate_width=True,
    extend_line=True)

# ---- img2: fluorescence vessels (light lines, branching) ----
# Exercises fix #5 (scale selection for light lines), #4 (junction remap after pruning)
run("img2_light_vessels",
    f"{IMG_DIR}/img2.jpg",
    line_widths=np.arange(3, 8),
    low_contrast=30, high_contrast=80,
    min_len=15, dark_line=False, estimate_width=True,
    extend_line=True)

# ---- img3: bright fibres on black background ----
# Exercises fix #5 (scale selection for light lines), #4 (junction remap after heavy pruning)
run("img3_light_fibres",
    f"{IMG_DIR}/img3.jpg",
    line_widths=np.arange(5, 11),
    low_contrast=80, high_contrast=160,
    min_len=20, dark_line=False, estimate_width=True,
    extend_line=False)

# ---- img4: spider web (light lines, many junctions) ----
# Exercises fix #1 (junction fallback), #4 (junction remap), #5 (light-line scale selection)
run("img4_light_junctions",
    f"{IMG_DIR}/img4.jpg",
    line_widths=np.arange(1, 4),
    low_contrast=50, high_contrast=120,
    min_len=10, dark_line=False, estimate_width=True,
    extend_line=True)

# ---- img5: bacteria (dark lines, noisy, grayscale 2D input) ----
# Exercises fix #3 (Taylor offset), #12 (negative coord clamp)
run("img5_dark_noisy",
    f"{IMG_DIR}/img5.png",
    line_widths=np.arange(2, 6),
    low_contrast=40, high_contrast=80,
    min_len=10, dark_line=True, estimate_width=True,
    extend_line=False)

# ---- img6: retinal vessels (light lines, multi-scale, branching) ----
# Exercises fix #5 (light-line scale selection across wide scale range), #4 (junction remap)
run("img6_retina_light",
    f"{IMG_DIR}/img6.jpg",
    line_widths=np.arange(1, 6),
    low_contrast=20, high_contrast=60,
    min_len=10, dark_line=False, estimate_width=True,
    extend_line=True)

# ---- img7: snake logo (dark lines, thick, width estimation + position correction) ----
# Exercises fix #9 (NaN guard in correct.py), #2 (normx sign), #3 (Taylor offset)
run("img7_dark_thick",
    f"{IMG_DIR}/img7.png",
    line_widths=np.arange(7, 11),
    low_contrast=50, high_contrast=100,
    min_len=15, dark_line=True, estimate_width=True,
    extend_line=True, correct_pos=True)

# ---- img4 grayscale as (H,W,1): test fix #11 (single-channel handling) ----
print(f"\n{'='*60}")
print("Test: img4_single_channel")
img = iio.imread(f"{IMG_DIR}/img4.jpg")
if img.ndim == 2:
    img = img[:, :, np.newaxis]
print(f"  Input shape: {img.shape} (artificially made H,W,1)")
run("img4_single_channel",
    img,
    line_widths=np.arange(1, 4),
    low_contrast=50, high_contrast=120,
    min_len=10, dark_line=False, estimate_width=True)

# ---- img1 with width estimation + empty-array guard (fix #18) ----
run("img1_width_guard",
    f"{IMG_DIR}/img1.jpg",
    line_widths=np.arange(1, 3),
    low_contrast=80, high_contrast=200,
    min_len=3, dark_line=True, estimate_width=True,
    extend_line=False)

print(f"\n{'='*60}")
print(f"All tests complete. Compare results in {OUT_DIR}/ with data/results/")
