# Setup

## Platform

Linux x86_64 with an NVIDIA GPU + CUDA. The SONIC controller (TensorRT/ONNX) and
curobo are built from source against CUDA; macOS/ARM is unsupported.

## Install

1. **Standard SIMPLE install** — follow the README **[Option 1] UV setup**:

   ```bash
   git submodule update --init --recursive
   uv sync --all-groups --index-strategy unsafe-best-match
   bash scripts/install_curobo.sh
   ```

2. **WBC extras** — one script adds what the standard install does not: the
   XRoboToolkit VR SDK build, the GR00T-WholeBodyControl controller and SONIC
   checkpoints, and the SONIC C++/TensorRT controller binary (it calls
   `build_sonic_controller.sh` as its last step).

   ```bash
   bash scripts/setup_teleop_wbc.sh
   ```

3. **XRoboToolkit PC Service** installed and running on the workstation wired to
   the headset.

## Required hardware

* **PICO 4 / PICO 4 Ultra** headset
* **2x PICO controllers**
* **2x PICO motion trackers** — strapped to the left and right ankles, indicator
  lights facing upward

## Headset configuration

1. Pair and calibrate the motion trackers in the PICO system settings, in
   **Full-body** mode.
2. Wear tight-fitting trousers — baggy clothing blocks tracker line-of-sight and
   causes erratic leg motion.
3. Open **XRoboToolkit** on the headset, connect to your PC's IP address, and
   confirm the status reads **`Working`**.
4. In XRoboToolkit settings:
   * Tracking: **Head** and **Controller**
   * Motion Tracker: **Full-body**
   * Remote Vision: **Zedmini** (Listen enabled for visual feedback)

For step-by-step screenshots of the PICO pairing, app install and calibration
flow, see [Teleop Setup](../decoupled-wbc/teleop_decoupled_setup.md) — the
headset-side setup is shared between the two whole-body paths.
