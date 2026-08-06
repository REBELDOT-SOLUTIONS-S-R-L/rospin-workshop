# ROSPin SO-101 workshop

A portable, browser-operated robotics workshop stack:

- MuJoCo physics with a custom Gymnasium environment
- keyboard teleoperation
- fixed top and wrist camera observations plus a movable perspective viewer
- local LeRobot v3.0 episode recording
- participant-authored Python trajectories for synthetic demonstrations
- local ACT policy training through `lerobot-train`
- one Docker workflow for Windows and macOS participants

The browser is the only rendering and control interface. MuJoCo runs headlessly
in Docker with OSMesa and streams both cameras to the page. Windows machines do
not need an X server or WSLg, macOS machines do not need XQuartz, and neither
platform needs Python or a native MuJoCo installation.

## Start on macOS

Install and start Docker Desktop, then run:

```bash
docker compose up --build
```

Open <http://localhost:8000>. Apple silicon uses the native `linux/arm64`
image; Intel Macs use `linux/amd64`.

The convenience script performs the same operation and follows the logs:

```bash
./scripts/workshop.sh start
```

Stop the stack with `docker compose down`. Recorded data remains under
`data/datasets/` on the host.

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
| `W` / `S` | end effector −Y / +Y |
| `A` / `D` | end effector +X / −X |
| `Q` / `E` | end effector +Z / −Z |
| `I` / `K` | wrist flex + / − |
| `J` / `L` | wrist roll + / − |
| `U` / `O` | shoulder pan + / − |
| `R` / `F` | shoulder lift + / − |
| `T` / `G` | elbow flex + / − |
| `[` / `]` | close fully / open fully |

The Gymnasium action is
`[eef_dx, eef_dy, eef_dz, shoulder_pan_delta, shoulder_lift_delta,`
`elbow_flex_delta, wrist_flex_delta, wrist_roll_delta, gripper_command]` in
`[-1, 1]`. Damped least-squares IK maps only the world-frame translation
command to the five arm joints. Rotation buttons address the named joint
directly, so a direct-joint command cannot be redistributed across the whole
arm. This follows LeIsaac's separation of Cartesian IK and direct SO-101 joint
control while retaining the requested three-axis Cartesian translation keys.

The robot mount faces 90 degrees toward world −Y, into the usable half of the
table. The perspective viewer starts behind it on the +Y side, offset to keep
the relocated robot and task objects in frame and aimed down at the workspace.
Drag the perspective image to orbit,
Shift-drag to pan, and use the mouse wheel or `+` / `−` buttons to zoom. Double
click or select **Reset view** to return to the starting pose. The perspective
camera is viewer-only: changing its pose never enters a recording.

The camera workspace occupies the top of the page, with equally sized 640×480
perspective and wrist views. The teleoperation, local-recording, and
session-status cards sit below the camera workspace without internal scrolling.
The wrist image and a fixed copy of the default perspective pose are recorded
as `observation.images.wrist` and `observation.images.top`. Moving the browser
perspective camera does not move the recorded top camera.

Full-scale held keys command at most 12 cm/s translation or 0.8 rad/s joint
motion, independent of the configured control rate. Releasing all arm-control
keys synchronizes position targets to the current pose so the robot stops
instead of finishing a queued motion. Gripper commands are absolute and
latched: one press targets the corresponding joint limit and release does not
interrupt the full open/close motion. The target is not shortened for a
particular object size; an explicit 0.08 N·m actuator-force cap lets contact
stop the jaw gently at the width of the object being grasped.

## Select a workshop task

Open the cube task directly at
<http://localhost:8000/?task=cube_in_bowl>. When only one task is installed,
opening the page without a query parameter selects it automatically. The first
selected task is locked for the server session; restart the container before
using a different task.

For a single-task workshop, server startup eagerly creates the simulation and
both camera renderers before logging `Application startup complete`. An already
open browser tab also reclaims its task and restarts both streams automatically
after a container restart; no manual refresh is required.

Task definitions live under `tasks/` and are mounted read-only at
`/workspace/tasks`. A task YAML selects reusable object catalogue entries,
places their instances in world coordinates, optionally gives dynamic objects
an X/Y spawn range, and defines its success and timeout rules. The cube is
sampled on every reset from X `[−0.15, 0.03]` and Y `[0.07, 0.20]`, with its
height fixed on the tabletop. Restarting the container reloads YAML changes
without rebuilding the image. The cube task automatically saves after the
released, settled cube remains fully inside the bowl for two seconds. A
20-second timeout discards the attempt. Both outcomes reset the scene and wait
for another **Start episode**. Manual save and discard remain available.

Task YAML cannot contain arbitrary MJCF or file paths. A new task needs only a
YAML file when all required catalogue objects and success predicates already
exist; a genuinely new mechanism first needs a reusable catalogue entry.

## Write and generate synthetic trajectories

Task YAML remains responsible for the scene, randomized object placement, and
success predicate. Participant behavior lives in ordinary Python files under
`trajectories/`. That directory is mounted read-only into the running
container, so editing a trajectory never requires rebuilding or restarting the
app.

Copy `trajectories/template.py` and define one decorated function:

```python
from rospin_workshop.trajectory import EpisodeContext, trajectory


@trajectory(task="cube_in_bowl")
def run(ctx: EpisodeContext) -> None:
    cube = ctx.object_position("cube")
    bowl = ctx.object_position("bowl")

    ctx.open_gripper()
    ctx.move_to(cube + [0.012, 0.0, 0.076], name="approach")
    ctx.move_linear(
        cube + [0.012, 0.0, -0.004],
        allow_contact=True,
        name="grasp",
    )
    ctx.close_gripper(until_contact=True)
    ctx.move_relative(z=0.10, name="lift")
    ctx.move_to(bowl + [0.0, 0.0, 0.13], name="transfer")
    ctx.open_gripper()
    ctx.move_relative(z=0.10, name="retreat")
    ctx.move_home()
```

Every phase starts from the measured robot state. `move_to()` uses a
vertical-horizontal-vertical tabletop-safe path, `move_linear()` follows a
straight Cartesian segment, and `move_relative()` resolves its endpoint when
the phase begins. Joint-space movement, force-limited gripper commands, waits,
settling checks, assertions, seeded randomness through `ctx.rng`, and
`move_home()` are also available. Intended grasp or placement contact must be
explicit with `allow_contact=True`; free-space phases fail if they cannot reach
their target.

The SO-101 has five arm joints and therefore cannot satisfy an arbitrary
six-dimensional XYZ-plus-quaternion goal. The initial API plans XYZ while
maintaining a continuous nearby arm posture. Participants can use
`move_joints()` when a particular arm orientation is required.

With the workshop app running, preview one seeded execution in the existing
browser viewer without recording:

```bash
docker compose exec -T workshop \
  rospin-generate cube_in_bowl.py --preview --seed 13
```

Generate a finalized dataset:

```bash
docker compose exec -T workshop \
  rospin-generate cube_in_bowl.py \
  --episodes 100 --seed 1000 --dataset-name team_cube
```

The default batch runner preflights each seeded episode without recording,
resets with the same seed, and records only after the program has proved it can
complete. During recording, task auto-save is deferred until the complete
program—including retreat and return home—has run. The YAML success predicate
then decides whether the episode is saved or discarded. Use `--no-preflight`
only when generation speed matters more than rejecting failures before video
encoding.

The convenience wrappers provide the same operations:

```bash
./scripts/workshop.sh preview cube_in_bowl.py 1 13
./scripts/workshop.sh generate cube_in_bowl.py 100 1000
```

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
| `observation.state` | `(6,)` | measured SO-101 positions in real-robot units |
| `observation.images.top` | `(H, W, 3)` | fixed top RGB, AV1 video |
| `observation.images.wrist` | `(H, W, 3)` | wrist RGB, AV1 video |
| `action` | `(6,)` | commanded SO-101 targets in real-robot units |

Physics and keyboard input run at 60 Hz independently from three camera-specific
render workers. The fixed top and wrist workers render the same simulation
snapshot and are recorded together at 25 FPS. The movable perspective worker
remains independent, so a slow browser view cannot delay dataset capture. Each
saved row combines state, action, and synchronized 640×480 top and wrist frames.
Keyboard and synthetic planner commands are both converted to the same six
absolute joint targets before recording. The action is the target sent to the
simulated motors, while `observation.state` is the six measured joint positions
at that snapshot. For example, if a joint is currently at 180° and a keyboard
step requests 181°, the stored observation is 180 and the action is 181—not the
1° delta. Key transitions do not inject extra off-cadence rows. Both scene
lights have shadows disabled and glossy material reflections are removed
because their redundant software-rendering passes prevent reliable 25 Hz
capture without helping workshop control.

MuJoCo continues to use radians internally. At recording time, the five arm
joints are converted to the calibrated degree convention used by the real
dataset. This includes inverting the source task's `−π/2` wrist-roll frame
correction. The gripper is mapped from simulated `[-10°, 100°]` to calibrated
`[0, 100]`. Both feature name lists use the real dataset's `shoulder_pan.pos`
through `gripper.pos` convention, and the dataset declares
`robot_type: so_follower`. The stored schema therefore matches
`/home/roboticslab/datasets/single_so101_cubes` and does not include
simulator-only velocity or end-effector features.

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
episodes, wrist camera stream, joint state, and action features.

The image build also runs `rospin-self-check`. It compiles and renders the
MuJoCo model, steps the EEF controller, records a temporary wrist-camera v3
episode, finalizes it, and decodes the video. A broken runtime therefore fails
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
URDF. Robot contact uses generated 2 mm boxes that are fully contained inside
each rendered STL solid; this preserves concave gaps without allowing a hidden
collision proxy to extend outside the visible robot. The cube-task visuals are
derived from the exact `cube_green.usd` and `Oala cuburi.usd` source meshes
under the IsaacTasks `assets/objects` directory. Because MuJoCo does not load
USD directly, deterministic OBJ copies and their source hashes are checked in
under `assets/objects/`; regenerate or verify them with:

```bash
python tools/convert_task_usd_objects.py
python tools/convert_task_usd_objects.py --check
```

The cube contact box matches its 25 mm visual volume exactly. The bowl uses
inset cylindrical floor and wall proxies that remain inside its tapered visual
shell.
The ground and lights come from the scene USD; the table is centred at the
scene origin, measures 0.75 m by 0.75 m, and renders neutral white. The SO-101's
printed parts use `#F88B17`. The checked-in robot collision
catalogue can be regenerated with `tools/generate_collision_boxes.py` after
installing its offline geometry dependencies.
URDF `rpy` values are extrinsic rotations, so the MJCF compiler deliberately
uses uppercase `XYZ`; lowercase `xyz` is intrinsic and produces incorrect link
and mesh poses. The verifier compiles the original URDF with MuJoCo's native
importer and compares all link and visual transforms at three joint poses.
The new URDF has no camera, so the wrist camera starts from the supplied camera
USD orientation. Its final pose and 62° vertical field of view were calibrated
against the physical `/dev/video2` MJPEG feed at 1280×720. The workshop UI and
every newly recorded wrist-camera video use the requested 640×480 resolution.

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
| `ROSPIN_CONTROL_HZ` | `60` | simulation physics and keyboard command rate |
| `ROSPIN_CAMERA_HZ` | `25` | synchronized top/wrist dataset FPS; perspective is best-effort |
| `ROSPIN_IMAGE_WIDTH` | `640` | wrist preview and recorded-video width |
| `ROSPIN_IMAGE_HEIGHT` | `480` | wrist preview and recorded-video height |
| `ROSPIN_DATA_ROOT` | `/workspace/data` | datasets and training outputs |
| `ROSPIN_TASKS_DIR` | `/workspace/tasks` | read-only task YAML directory |
| `ROSPIN_TRAJECTORIES_DIR` | `/workspace/trajectories` | hot-loaded participant Python directory |
| `ROSPIN_SO101_ASSET_DIR` | auto-detected | vendored SO-101 URDF directory |
| `MUJOCO_GL` | `osmesa` | CPU headless OpenGL backend |

The workshop wrist-video schema is 640×480. Changing these values creates a
different dataset schema; do not change them midway through a dataset.

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
