RBE/CS 549 Computer Vision - Project 3: Einstein Vision: A Monocular Perception and 3D Scene Reconstruction Pipeline for Autonomous Driving Visualization
Author: Filippo Marcantoni (fmarcantoni@wpi.edu), Prahladh Raja (pnamboorkrishnam@wpi.edu) (Group 10)
Course: RBE/CS 549 - Computer Vision (Prof. Nitin Sanket)
Institution: Worcester Polytechnic Institute

-------------------------------------------------------------------------------
Overview
-------------------------------------------------------------------------------

This project implements a complete monocular autonomous-driving perception and
visualization pipeline. Starting from raw driving video, the system performs
object detection, traffic-light detection, lane and road understanding, monocular
depth estimation, vehicle 3-D localization, optical-flow-based motion reasoning,
scene assembly, and a final Blender rendering.

The overall goal is to reconstruct a Tesla-style multi-view simulation from
real road scenes by combining learning-based perception with geometric scene
reasoning and asset-based rendering.

The pipeline is organized as a sequence of stages that can be run independently
or composed together depending on the desired output:

1. 2-D object detection for vehicles, pedestrians, traffic lights, and road signs
2. Lane and road / freespace understanding
3. Monocular metric depth estimation
4. Vehicle 3-D bounding-box estimation and orientation recovery
5. Optical-flow-based motion estimation and temporal stabilization
6. Scene assembly into a structured JSON scene representation
7. Blender-based scene recreation and final rendering

-------------------------------------------------------------------------------
Pipeline Summary
-------------------------------------------------------------------------------

Input:
- Front-camera monocular driving video
- Optional precomputed detection JSON files
- Optional precomputed depth maps
- Dataset assets for Blender rendering

Core perception stages:
- Object detection
- Traffic-light recognition
- Lane / road segmentation
- Metric monocular depth estimation
- Vehicle 3-D detection
- Optical flow and motion reasoning

Output:
- Annotated detection videos
- Depth visualization videos and depth_maps.npz
- Vehicle 3-D detection JSONs and overlays
- Lane / traffic-light / motion outputs
- Scene assembly JSON
- Final Blender-rendered multi-view driving simulation

-------------------------------------------------------------------------------
Installation and Environment Setup
-------------------------------------------------------------------------------

Recommended Python version:
- Python 3.10 or 3.11

A virtual environment is strongly recommended.

1) Create and activate a virtual environment

Linux / macOS:
    python3 -m venv .venv
    source .venv/bin/activate

Windows PowerShell:
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

2) Upgrade pip

    python -m pip install --upgrade pip setuptools wheel

3) Install core Python dependencies

    pip install numpy opencv-python pillow scipy matplotlib tqdm imageio scikit-image scikit-learn pandas pyyaml requests

4) Install PyTorch and vision packages

For a standard pip install:
    pip install torch torchvision torchaudio

For CUDA-enabled PyTorch, install the correct wheel for the target machine.
Example:
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

5) Install model/runtime dependencies

    pip install ultralytics timm einops transformers

Optional fallback detector:
    pip install groundingdino-py

6) Blender requirements

Blender is required for the final rendering stage.
Install Blender separately from the official Blender distribution and run the
rendering script through Blender’s Python runtime, for example:

    blender --background --python blender_renderer.py -- [args...]

Inside Blender, the project uses:
- bpy
- mathutils

These are provided by Blender’s embedded Python and are not installed through pip
for standard system Python use.

-------------------------------------------------------------------------------
External Repositories and Models
-------------------------------------------------------------------------------

Several parts of the pipeline depend on external repositories or pretrained models.

Create an external folder first:

    mkdir -p external

Then clone the required repositories.

1) Ford Otosan lane / segmentation repository

    cd external
    git clone https://github.com/recepayddogdu/Object_DetectionClassification-_Ford_Otosan_Intern_P2.git

Used by:
- line_SegNet.py
- lane_detection.py

Purpose:
- SegNet-based lane segmentation
- lane marking extraction

2) Ford Otosan freespace segmentation repository

    git clone https://github.com/recepayddogdu/Freespace_Segmentation-Ford_Otosan_Intern.git

Purpose:
- road and freespace segmentation
- road-area understanding in the lane / road reconstruction stage

3) skhadem 3D BoundingBox repository

    git clone https://github.com/skhadem/3D-BoundingBox.git

Purpose:
- DeepBox-style 3-D vehicle box estimation backend
- used through local wrapper / adapter code

If using this backend, download its weights:

    cd 3D-BoundingBox/weights
    chmod +x get_weights.sh
    ./get_weights.sh
    cd ../../

Note:
Some older weight-download links may fail. In that case, either provide the
checkpoint manually or use the project’s built-in geometry fallback backend.

4) Optional additional 3-D box repositories

    git clone https://github.com/GhaziXX/3d-bounding-box-detection.git

This can be used as an additional reference / experimental backend, though the
main integrated backend in this project is the skhadem implementation plus local
geometry fitting wrappers.

5) Optional DETIC

DETIC can be installed separately if used for object detection, subclassification,
or brake-light detection.

Reference:
    https://github.com/facebookresearch/Detic

6) Optional Mask R-CNN and DETR references

These are reference implementations used for experiments / fallback directions:
- Lane / road marks with Mask R-CNN
- Traffic sign detection with DETR

-------------------------------------------------------------------------------
Required Local Helper Files
-------------------------------------------------------------------------------

Some external backends are wrapped by local project files. These files must remain
in the project root (or a visible code/ folder on PYTHONPATH) for the pipeline to work:

- project_setup.py
- deepbox_geometry.py
- skhadem_deepbox.py
- cersar_3d_detection.py
- lzccccc_3dbox.py
- vehicle_subclassification.py
- calibration.py

These are project modules, not pip packages.

-------------------------------------------------------------------------------
Package / Dependency Reference
-------------------------------------------------------------------------------

Core packages used across the project:

- numpy
  Used across the pipeline for array operations, geometry, depth-map processing,
  and numeric scene data handling.

- opencv-python (cv2)
  Used for video I/O, image preprocessing, classical vision operations, drawing
  overlays, and fallback optical flow.

- torch
  Used as the main runtime for neural-network inference in detection, depth
  estimation, lane segmentation, and optical flow.

- torchvision
  Used in the optical-flow stage to run the RAFT backend.

- ultralytics
  Used to run YOLO-based object detection and traffic-light proposals.

- groundingdino
  Used as an optional fallback detector in the object-detection stage.

- bpy
  Used to construct, animate, and render the final 3-D scene in Blender.

- mathutils
  Used inside Blender for vectors, matrices, rotations, and transform handling.

-------------------------------------------------------------------------------
Repository / Model Reference
-------------------------------------------------------------------------------

- Object_DetectionClassification-_Ford_Otosan_Intern_P2
  Vendored Ford Otosan repository used for the SegNet-based lane-segmentation
  model wrapped by line_SegNet.py and lane_detection.py.

- Freespace_Segmentation-Ford_Otosan_Intern
  Vendored Ford Otosan repository used for road and freespace segmentation in
  the lane / road reconstruction stage.

- Ultralytics YOLOv8
  Main detector family used for vehicles, pedestrians, traffic lights, and
  road-sign proposals.

- Grounding DINO
  Optional open-vocabulary fallback used when YOLO misses target classes.

- RAFT
  Dense optical-flow model used to estimate inter-frame motion for orientation
  stabilization.

- ZoeDepth
  Preferred monocular depth model because it predicts metric depth directly.

- MiDaS
  Depth-estimation fallback used when ZoeDepth is unavailable.

- Blender
  Final rendering platform used to rebuild the assembled scene with dataset
  assets and export the simulation.

- DETIC
  Used for object detection, car subclassification, and brake-light detection.
  Reference:
  https://github.com/facebookresearch/Detic

- 3D Bounding Box - DeepBox
  Used for vehicle 3-D position and orientation detection.
  Reference:
  https://github.com/skhadem/3D-BoundingBox

- Mask R-CNN
  Used for lane and road-mark detection experiments.
  Reference:
  https://debuggercafe.com/lane-detection-using-mask-rcnn/

- DETR
  Fallback for traffic-sign detection.
  Reference:
  https://debuggercafe.com/traffic-sign-detection-using-detr/

-------------------------------------------------------------------------------
How to Run
-------------------------------------------------------------------------------

Below are representative examples. Paths should be adapted to the local machine
or cluster environment.

1) Object detection

    python object_detection.py \
      --video P3Data/Sequences/scene10/Undist/<video_name>.mp4 \
      --scene scene10 \
      --device auto \
      --out-video output/scene10/detections/detections_output.mp4 \
      --out-json output/scene10/detections/detections.json

2) Depth estimation

    python depth_estimation.py \
      --video P3Data/Sequences/scene10/Undist/<video_name>.mp4 \
      --scene scene10 \
      --device auto \
      --model-type auto \
      --out output/scene10/depth/depth_output.mp4 \
      --out-npz output/scene10/depth/depth_maps.npz

3) Vehicle 3-D detection using detections.json and depth maps

    python vehicle_3d_detection.py \
      --scene scene10 \
      --view front \
      --video P3Data/Sequences/scene10/Undist/<video_name>.mp4 \
      --device auto \
      --detections-json output/scene10/detections/detections.json \
      --depth-npz output/scene10/depth/depth_maps.npz \
      --deepbox-backend auto \
      --out-video output/scene10/detections/vehicle_3d_output.mp4 \
      --out-json output/scene10/detections/vehicle_3d_detections.json

4) Lane detection / segmentation
   Run according to the selected backend and local model checkpoints.

5) Optical flow / motion estimation

    python optical_flow.py [args...]

6) Scene assembly

    python scene_assembler.py [args...]

7) Blender rendering

    blender --background --python blender_renderer.py -- [args...]

-------------------------------------------------------------------------------
Notes on Cluster / Turing Usage
-------------------------------------------------------------------------------

When running on a cluster:
- request GPU resources before launching the scripts
- ensure PyTorch is installed with a CUDA-compatible wheel for the node
- verify GPU visibility with:
      python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
- if CUDA is unavailable, the scripts may silently fall back to CPU

For multi-GPU usage with the current scripts:
- one process typically uses one GPU
- the practical pattern is to run different scenes on different GPUs, e.g.
      CUDA_VISIBLE_DEVICES=0 python depth_estimation.py ...
      CUDA_VISIBLE_DEVICES=1 python depth_estimation.py ...


