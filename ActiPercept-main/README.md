# ActiPercept - Active Perception for Screw Detection 

This repository contains an active perception tool, called **ActiPercept** that enables the control of a camera in order to
**maximize screw detection confidence**.

------------------------------------------------------------------------

## 🚀 Before You Start

### 1️⃣ Export Webots environment variables (required for external controllers)

``` bash
export WEBOTS_HOME=/usr/local/webots
```

Make sure this matches your local Webots installation path.

------------------------------------------------------------------------

### 2️⃣ Open the Webots world

Start Webots and open:

    worlds/world.wbt

------------------------------------------------------------------------

### 3️⃣ Run the external controller

From the repository root:

``` bash
$WEBOTS_HOME/webots-controller/ ./demo_keyboard.py
```

This launches the keyboard-based demo controller.

------------------------------------------------------------------------

## 📂 Repository Structure

    repo/
    │
    ├── Controllers/
    │   ├── demo_keyboard.py
    │   ├── screw_detector_env.py
    │   ├── utils_vision.py
    │   └── pc_screw_detection.pt
    │
    ├── protos/
    │   └── (custom Webots PROTO files and textures)
    │
    ├── worlds/
    │   └── world.wbt
    │
    └── README.md

------------------------------------------------------------------------

## 🧠 How It Works

1.  YOLO runs on every camera frame (`screw`, `noscrew`, optionally
    `holder`).
2.  Each detection is assigned to one of four corners:
    -   Top-Right (TR)
    -   Bottom-Right (BR)
    -   Top-Left (TL)
    -   Bottom-Left (BL)
3.  When a corner is selected (keys 1--4), the controller enters
    **refine mode**:
    -   Prefer tracking `holder` in that corner (short TTL cache).
    -   Otherwise track `screw` / `noscrew`.
    -   Center the detection in the image.
    -   Move closer until confidence ≥ threshold.
    -   Return to home pose.

------------------------------------------------------------------------

## 🎮 Controls

### Free Motion

-   **W / A / S / D** → Move in X/Y plane\
-   **Q / E** → Move down / up (Z)

### Select Corner (Enter Refine Mode)

-   **1** → Top-Left\
-   **2** → Top-Right\
-   **3** → Bottom-Right\
-   **4** → Bottom-Left

------------------------------------------------------------------------

## ⚙ Key Parameters (tune in `ScrewDetectorEnv`)

-   `conf_threshold` (default `0.85`)\
    Confidence required to finish refinement.

-   `center_epsilon_px` (default `12`)\
    Pixel tolerance used for deciding if the object is centered.

-   `holder_cache_ttl` (default `5`)\
    Number of steps to keep a cached holder if it temporarily
    disappears.

-   `move_step` (default `0.01`)\
    Translation step per action.

-   `workspace_bounds`\
    Safety bounds `[minx,maxx,miny,maxy,minz,maxz]` for the camera
    workspace.


## Acknowledgments

The work presented here is supported by the RoboSAPIENS project funded by the European Commission’s Horizon Europe programme under grant agreement number 101133807.
This publication reflects the authors’ views only. The European Commission is not responsible for any use that may be made of the information it contains.

<p align="center"> <img src="https://raw.githubusercontent.com/cidl-auth/mitigate-anomalies/main/assets/eu_funded.jpg" alt="Funded by the European Union" width="180"/> </p> <p align="center"> Learn more about <a href="https://robosapiens-eu.tech/" target="_blank"><strong>RoboSAPIENS</strong></a>. </p> <p align="center"> <img src="https://raw.githubusercontent.com/cidl-auth/mitigate-anomalies/main/assets/robosapiens_robot.png" alt="RoboSAPIENS" width="80"/> </p>
