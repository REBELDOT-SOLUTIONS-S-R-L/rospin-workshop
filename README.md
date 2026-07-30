# ROSPin SO-101 workshop

A portable, browser-operated robotics workshop stack:

- MuJoCo physics with a custom Gymnasium environment
- 6-DoF end-effector keyboard teleoperation
- wrist and fixed perspective camera observations
- local LeRobot v3.0 episode recording
- local ACT policy training through `lerobot-train`
- one Docker workflow for Windows participants

The browser is the user interface, so Windows machines do not need an X server,
WSLg, Python, or a native MuJoCo installation.

## Start on Windows

Prerequisites:

1. Install Docker Desktop.
2. Select **Linux containers** and enable the WSL 2 backend.
3. Allocate at least 8 GB RAM to Docker; 12–16 GB is preferable when training.

From PowerShell in this directory:

```powershell
docker compose up --build
```

Open <http://localhost:8000>. Docker selects the laptop's native Linux platform
(`linux/amd64` on Intel/AMD Windows laptops and `linux/arm64` on supported
Windows-on-ARM machines).

The convenience script performs the same operation and follows the logs:

```powershell
.\scripts\workshop.ps1 start
```

Stop the stack with `docker compose down`. Recorded data remains under
`data/datasets/` on the host.

## Teleoperation

The page focuses the keyboard-control panel when it connects. After editing a
dataset or task field, click the highlighted control panel again, then hold:

| Keys | Command |
|---|---|
| `W` / `S` | end effector +X / −X |
| `A` / `D` | end effector +Y / −Y |
| `Q` / `E` | end effector +Z / −Z |
| `I` / `K` | roll about world +X / −X |
| `J` / `L` | pitch about world +Y / −Y |
| `U` / `O` | yaw about world +Z / −Z |
| `[` / `]` | close / open gripper |

The Gymnasium action is
`[eef_dx, eef_dy, eef_dz, eef_droll, eef_dpitch, eef_dyaw, gripper_delta]`
in `[-1, 1]`. A weighted damped least-squares Jacobian controller maps
world-frame translation and rotation commands to the five arm joints. Because
SO-101 has five arm degrees of freedom, arbitrary six-dimensional poses are
underactuated; the controller produces the closest differential motion and
prioritizes translation. MuJoCo position actuators execute the resulting joint
targets. Full-scale held keys command at most 8 cm/s translation or 0.5 rad/s
rotation, independent of the configured control rate.

## Record a local LeRobot v3 dataset

The recording controls deliberately separate episode and dataset boundaries:

1. Enter a dataset name and a task description. Until workshop tasks are
   defined, the default description is valid.
2. Select **Start episode**, teleoperate, then select **Save episode**.
3. Repeat step 2 to add demonstrations to the same dataset.
4. Select **Finish dataset**. This closes video encoders and writes Parquet
   footers; do this before validation or training.
5. Use **Discard** instead of **Save episode** to reject the current take.

Closing the container also attempts to save an active episode and finalize its
dataset. Explicitly using **Finish dataset** is safer.

Every frame contains:

| LeRobot feature | Shape | Meaning |
|---|---:|---|
| `observation.state` | `(6,)` | six joint positions |
| `observation.velocity` | `(6,)` | six joint velocities |
| `observation.eef_position` | `(3,)` | end-effector xyz |
| `observation.eef_orientation` | `(4,)` | end-effector quaternion, wxyz |
| `observation.images.wrist` | `(H, W, 3)` | wrist RGB, H.264 video |
| `observation.images.perspective` | `(H, W, 3)` | fixed RGB, H.264 video |
| `action` | `(7,)` | translation, rotation, and gripper command |

Physics and keyboard input run at 20 Hz independently from the two-camera
renderer. The original STL meshes are detailed enough that portable CPU
software rendering is configured at 5 FPS; recordings use that same camera
rate so their timestamps and H.264 streams remain consistent.

Datasets are never pushed to Hugging Face Hub. Because v3 metadata does not
serialize `repo_id`, the tools deterministically use `local/<dataset-directory>`
when opening a dataset.

Validate a finalized recording:

```powershell
Get-ChildItem .\data\datasets
.\scripts\workshop.ps1 check <dataset-directory>
```

Equivalent direct command:

```powershell
docker compose run --rm --entrypoint rospin-check-dataset workshop `
  /workspace/data/datasets/<dataset-directory>
```

The validator loads the actual dataset and checks its version, frames,
episodes, camera streams, joint state, and action features.

The image build also runs `rospin-self-check`. It compiles and renders the
MuJoCo model, steps the EEF controller, records a temporary two-camera v3
episode, finalizes it, and decodes both videos. A broken runtime therefore fails
during `docker compose up --build`, before the workshop begins.

## Train ACT locally

CPU training is supported everywhere but will be slow:

```powershell
.\scripts\workshop.ps1 train <dataset-directory> -Steps 100000
```

Checkpoints are written to `data/outputs/act_<dataset-directory>/`. The wrapper
sets:

- `--policy.type=act`
- the dataset's local root and metadata `repo_id`
- the PyAV decoder used by the recording stack
- `--policy.push_to_hub=false`
- `--wandb.enable=false`

To avoid downloading pretrained ResNet weights, use the direct command and add
`--no-pretrained-backbone`.

### Optional NVIDIA GPU

Docker Desktop must already expose the NVIDIA GPU to Linux containers. Build
and run the CUDA variant:

```powershell
docker compose -f compose.yaml -f compose.gpu.yaml up --build
.\scripts\workshop.ps1 train <dataset-directory> -Gpu -Steps 100000
```

The CPU and CUDA images are intentionally separate so the default workshop
download stays hardware-neutral.

The `Docker image` CI workflow builds and starts the CPU image independently on
native `amd64` and `arm64` GitHub runners. Each job runs both the Dockerfile
self-check and an HTTP runtime smoke test.

## Source assets and conversion

The three-robot
[`lerobot/robot-urdfs`](https://huggingface.co/buckets/lerobot/robot-urdfs)
bucket is synchronized locally at `/home/roboticslab/assets/robot-urdfs`.
Hugging Face Storage Buckets do not support `git clone`; their server-prescribed
equivalent is:

```bash
hf sync hf://buckets/lerobot/robot-urdfs \
  /home/roboticslab/assets/robot-urdfs
```

The bucket contains `g1/`, `openarm/`, and `so101/`. An exact copy of `so101/`
is vendored in this project at `assets/robots/so101/`.

MuJoCo loads the checked-in MJCF scene at
`src/rospin_workshop/models/so101_workshop.xml`. Its robot is built from:

- `assets/robots/so101/so101_new_calib.urdf`
- all 13 STL files under `assets/robots/so101/assets/`
- `assets/scenes/scene.usd`

All 17 URDF visual instances use their real STL mesh, material, link, and
origin transform. Joint frames, axes, inertials, and limits also come from the
URDF. Only invisible collision primitives are simplified for stable,
lightweight CPU contact. The ground, table, and lights come from the scene USD.
URDF `rpy` values are extrinsic rotations, so the MJCF compiler deliberately
uses uppercase `XYZ`; lowercase `xyz` is intrinsic and produces incorrect link
and mesh poses. The verifier compiles the original URDF with MuJoCo's native
importer and compares all link and visual transforms at three joint poses.
The new URDF has no camera, so the wrist camera retains the orientation from
the supplied camera USD and uses an 8 cm side bracket to keep its lens clear of
the real gripper mesh.

To verify source/model synchronization:

```bash
python -m venv .venv
.venv/bin/pip install '.[dev]'
.venv/bin/python tools/verify_asset_conversion.py
```

The manifest used by this check is
`src/rospin_workshop/models/asset_manifest.json`. Original assets are also
copied into the Docker image under `/assets`.

## Configuration

Set environment variables in `compose.yaml`:

| Variable | Default | Purpose |
|---|---:|---|
| `ROSPIN_CONTROL_HZ` | `20` | simulation physics and keyboard command rate |
| `ROSPIN_CAMERA_HZ` | `5` | browser camera and dataset recording FPS |
| `ROSPIN_IMAGE_WIDTH` | `320` | both camera widths |
| `ROSPIN_IMAGE_HEIGHT` | `240` | both camera heights |
| `ROSPIN_DATA_ROOT` | `/workspace/data` | datasets and training outputs |
| `ROSPIN_SO101_ASSET_DIR` | auto-detected | vendored SO-101 URDF directory |
| `MUJOCO_GL` | `osmesa` | CPU headless OpenGL backend |

Both ACT camera inputs must have the same resolution. Changing resolution
creates a different dataset schema; do not change it midway through a dataset.

## Local development and tests

Python 3.12 is required.

```bash
python -m venv .venv
.venv/bin/pip install '.[dev]'
MUJOCO_GL=egl .venv/bin/pytest
```

The environment is registered as `ROSpin/SO101Workshop-v0`, so local code may
use `gymnasium.make("ROSpin/SO101Workshop-v0")` after importing
`rospin_workshop`.

Use `MUJOCO_GL=osmesa` on systems with OSMesa installed. The Docker image uses
OSMesa by default and is the supported workshop runtime.
