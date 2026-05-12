# Droplet Platform

Acquisition and analysis software for a wide-field droplet microfluidic platform designed for parallel slow-readout fluorescence assays and post-measurement droplet recovery.

The repository is organized so that the main Jupyter notebook remains a thin user interface, while acquisition, detection, tracking, preview, video replay, video recording, ROI photometry and analysis logic live in importable Python modules.

## Core idea

In the experimental workflow, each droplet is treated as a computationally addressable measurement channel. A camera observes many droplets in one field of view, software detects and tracks each droplet, droplet-wise fluorescence traces are recorded, and a selected droplet can later be tracked toward a collection zone.

This architecture is useful for assays where the informative signal is slow or weak and cannot be measured reliably from a single fast point readout.

## Features

- Live acquisition from Basler cameras through `pypylon`.
- Fake streaming from prerecorded video files for demos and detector tuning.
- Live OpenCV preview with droplet detection/tracking overlays.
- Runtime detector-parameter tuning from the notebook UI.
- Raw-frame snapshots and overlay snapshots.
- Optional raw-video and overlay-video recording during preview.
- GPU-accelerated correlation detector using either direct correlation or FFT-based correlation.
- Optional enhanced two-pass detector wrapper.
- Torch neural-detector wrapper with the same public detector API.
- Identity-preserving droplet tracking.
- Droplet-wise ROI intensity recording into TSV tables.
- Basic MFE-style downstream analysis helpers.

## Repository layout

```text
src/droplet_platform/
  acquisition/       Basler camera, video replay, video recording
  detection/         correlation detector, enhancer, neural detector, common API
  tracking/          identity-preserving droplet tracker
  measurement/       ROI photometry and droplet-wise TSV recording
  ui/                notebook widgets, preview loop, callbacks, finish-zone logic
  analysis/          MFE processing and plotting helpers
  utils/             config and path utilities
notebooks/
  main_ui.ipynb      main acquisition/control notebook
  training_droplet_net.ipynb
  analysis_legacy.ipynb
configs/
  default.yaml
models/
  droplet_net.pt
legacy/
  original scripts/notebooks kept for traceability only
```

## Installation

Development install:

```bash
cd droplet_platform_repo
python -m pip install -e .
```

For Basler camera acquisition, install Basler pylon first and then:

```bash
python -m pip install pypylon
```

For GPU correlation detection, install the CuPy build matching your CUDA version, for example:

```bash
python -m pip install cupy-cuda12x
```

For the neural detector:

```bash
python -m pip install torch
```

## Main notebook

Open:

```text
notebooks/main_ui.ipynb
```

The notebook supports two source modes:

1. `Camera`: live Basler camera stream.
2. `Video file`: prerecorded video replay used as fake streaming for demos and detector tuning.

The UI allows adjustment of:

- FPS and exposure time;
- detector percentile threshold;
- image downsampling factor;
- droplet radius and minimum distance;
- direct vs FFT correlation backend;
- colorized preview on/off;
- detection and ID overlays;
- selected droplet ID and finish-zone geometry;
- raw/overlay video recording;
- droplet-wise intensity recording parameters.

## Droplet-wise intensity recording

The cleaned recording module is:

```python
from droplet_platform.measurement import record_droplet_intensities
```

It records a TSV table with one time column and one column per droplet ID:

```text
timestamp_ns    id0001    id0002    id0003 ...
...
```

Each `idXXXX` column contains the integrated mono8 intensity inside a circular ROI centered on the tracked droplet position. Missing positions are written as `nan`, not zero, so tracking gaps remain distinguishable from real low fluorescence.

## Detector API

All detectors are expected to implement:

```python
detect_centers(frame_gray, exclude_border=False, return_scores=False)
```

This keeps correlation, enhanced and neural detectors interchangeable in preview, tracking and recording code.

## Public-repository note

The active source code and notebooks do not contain user-specific absolute file paths. Example paths in configuration files are intentionally relative. Legacy notebooks are kept only for traceability and should not be used as the primary development surface.

## Development direction

New code should go into `src/droplet_platform/`. The notebook should remain a control panel and should not accumulate experimental logic.


## Debugging preview problems

The main UI writes status messages and full Python tracebacks into the notebook log box and also appends them to `ui_debug.log` inside the selected experiment root directory. If preview opens and immediately stops, check this log first. For video replay, the UI now validates that the selected file exists before starting the background preview thread.

OpenCV preview windows require the standard `opencv-python` package. Do not install `opencv-python-headless`, because it does not include HighGUI support (`cv2.namedWindow`, `cv2.imshow`).
