# Multi-Scale Ridge Detector

## Overview

This repository contains the Python implementation of a multi-scale ridge detector designed for identifying and measuring curvilinear structures in images across various fields including medical imaging and remote sensing. For more detailed information, please refer to the following literature:

[1] Steger, Carsten. "An unbiased detector of curvilinear structures." IEEE Transactions on Pattern Analysis and Machine Intelligence, 20(2), pp. 113-125, 1998.

[2] Lindeberg, Tony. "Edge detection and ridge detection with automatic scale selection." International Journal of Computer Vision, 30, pp. 117-156, 1998.

[3] Sato, Yoshinobu, et. al. "Three-dimensional multi-scale line filter for segmentation and visualization of curvilinear structures in medical images." Medical Image Analysis, 2(2), pp. 143-168, 1998.


This Python implementation offers a multi-scale adaptation of the ImageJ ridge detection [plugin](https://imagej.net/Ridge_Detection). The detection is more accuracy compared to the ImageJ plugin (in almost all cases we have tested :)) and faster than another Python implementation [ridge-detection](https://pypi.org/project/ridge-detection/). 

## Installation

### Prerequisites

- Python 3.8 or higher
- Opencv & scikit-image


### Install via pip:

   ```bash
   pip install ridge-detector
   ```
or

  ```bash
   pip install git+https://github.com/lxfhfut/ridge-detector.git
  ```

### Usage:
```python
from ridge_detector import RidgeDetector
det = RidgeDetector(line_widths=[1, 2, 3],  # Line widths to detect
                    low_contrast=100,  # Lower bound of intensity contrast
                    high_contrast=200,  # Higher bound of intensity contrast
                    min_len=10, # Ignore ridges shorter than this length
                    max_len=0, # Ignore ridges longer than this length, set to 0 for no limit
                    dark_line=True, # Set to True if detecting black ridges in white background, False otherwise
                    estimate_width=True, # Estimate width for each detected ridge point
                    extend_line=True, # Tend to preserve ridges near junctions if set to True
                    correct_pos=False,  # Correct ridge positions with asymmetric widths if set to True
                    )
det.detect_lines("path_to_image_or_image_data_in_numpy_array")
det.show_results()
det.save_results("directory_to_store_results")  # Comment out if you want to save the detection results
```

### Demonstrations

| Original                                                    | Ridges & Widths                                                              | Binary Ridges                                                                | Binary Widths                                                              |
|-------------------------------------------------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img0.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img0_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img0_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img0_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img1.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img1_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img1_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img1_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img2.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img2_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img2_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img2_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img3.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img3_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img3_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img3_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img4.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img4_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img4_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img4_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img5.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img5_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img5_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img5_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img6.jpg" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img6_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img6_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img6_binary_widths.png" width="200" height="200"> |
| <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/images/img7.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img7_contours_widths.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img7_binary_contours.png" width="200" height="200"> | <img src="https://github.com/lxfhfut/ridge-detector/raw/main/data/results/img7_binary_widths.png" width="200" height="200"> |
