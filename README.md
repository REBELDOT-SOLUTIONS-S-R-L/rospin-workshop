# ROSPIN SO-101 Workshop

This is a browser-based robotics workshop for the LeRobot SO-101 arm. It runs
a headless MuJoCo simulation in Docker and provides:

- keyboard teleoperation with Cartesian and direct-joint controls;
- a movable perspective view and a recorded wrist-camera view;
- the cube-in-bowl scene loaded from its YAML definition;
- local LeRobot v3 dataset recording;
- participant-written Python trajectories for synthetic demonstrations; and
- local ACT policy training.

No physical leader arm is used. Participants control the simulated follower
with the keyboard or with trajectory code. The generated motor values and
camera schema match the real SO-101 reference dataset at
`/home/roboticslab/datasets/single_so101_cubes`.

## Requirements

- Git
- Docker with Docker Compose v2
- at least 8 GB of RAM available to Docker; 12–16 GB is preferable for training
- a current Chrome, Edge, Firefox, or Safari browser

Python, MuJoCo, and LeRobot do not need to be installed on the host. They are
included in the Docker image.

## Install and run

Clone the repository first:

```bash
git clone https://github.com/REBELDOT-SOLUTIONS-S-R-L/rospin-workshop.git
cd rospin-workshop
```

### macOS

Install and start Docker Desktop. Both Apple silicon and Intel Macs are
supported. Then run:

```bash
./scripts/workshop.sh start
```

Open <http://localhost:8000>.

### Windows

Install Docker Desktop, use Linux containers, enable its WSL 2 backend, and
start Docker Desktop. From PowerShell in the repository:

```powershell
.\scripts\workshop.ps1 start
```

Open <http://localhost:8000>.

If script execution is disabled for the current PowerShell process, enable
local scripts temporarily:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

### Linux

Install Docker Engine and the Docker Compose plugin, then make sure the current
user can run `docker` commands. From the repository:

```bash
./scripts/workshop.sh start
```

Open <http://localhost:8000>.

### Stop the application

On macOS or Linux:

```bash
./scripts/workshop.sh stop
```

On Windows:

```powershell
.\scripts\workshop.ps1 stop
```

Recorded datasets and training outputs remain under `data/` on the host.
The workshop scripts pass this directory to Docker as an absolute host path.
At startup, Compose creates the named `rospin-workshop-data` volume as a
bind-backed volume whose storage device is that host directory. It therefore
appears in Docker Desktop's Volumes view while keeping datasets directly under
the repository's `data/` directory.

## Cube-in-bowl task

The workshop always loads `cube_in_bowl` when the server starts. The task asks
the robot to pick up the green cube and place it in the black bowl. Its scene,
random cube spawn range, success conditions, and 60-second timeout are defined
in [`tasks/cube_in_bowl.yaml`](tasks/cube_in_bowl.yaml).

## Use the browser UI

The header shows whether the page is connected. Wait for **Connected** and for
both camera feeds before starting an episode. The page reconnects to the
cube-in-bowl task automatically after an application restart.

The UI has four working areas:

1. **Perspective camera** — viewer only; it is never written to the dataset.
   Drag to orbit, Shift-drag to pan, use the wheel or `+`/`−` to zoom, and
   double-click or select **Reset view** to restore the default camera.
2. **Wrist camera** — the live 640 × 480 image recorded in the dataset.
3. **Keyboard teleoperation** — click this panel after editing the dataset-name
   field so that movement keys control the robot again.
4. **Local recording and session status** — starts, saves, discards, and
   finalizes episodes; displays the timer, frame count, task outcome, end
   effector pose, and dataset path.

**Reset simulation** restores the robot home pose and respawns the cube at a
randomized workspace position. It is disabled while an episode is being
recorded.

## Teleoperation

Hold multiple movement keys together for combined motion.

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
| `[` | close the gripper |
| `]` | open the gripper |

Translation keys command the end effector in world X/Y/Z through inverse
kinematics. Rotation keys command the named joint directly. Releasing the arm
keys stops motion at the current pose. Gripper commands are absolute and
latched; contact and the actuator force cap stop the fingers on objects of
different sizes without a fixed position limit.

## Record demonstrations

Use the **Local recording** panel:

1. Enter a dataset name. The cube-in-bowl description is filled automatically
   and cannot be edited in the UI.
2. Select **Start episode**.
3. Complete the task with the keyboard.
4. Select **Save episode** to keep the take, or **Discard** to reject it.
5. Repeat steps 2–4 for additional demonstrations.
6. Select **Finish dataset** when done. This closes the video encoders and
   writes the final metadata and Parquet files.

For the cube task, satisfying all YAML success conditions for two seconds
automatically saves the episode. Reaching the 60-second timeout automatically
discards it. Always use **Finish dataset** before validation, copying, merging,
or training.

Each recording is created at:

```text
data/datasets/<dataset-name>_<UTC-timestamp>/
```

Validate a finalized dataset on macOS or Linux:

```bash
./scripts/workshop.sh check <dataset-directory>
```

On Windows:

```powershell
.\scripts\workshop.ps1 check <dataset-directory>
```

Pass the directory name shown under `data/datasets/`, not its complete host
path.

## Dataset format

The workshop application writes a local LeRobot v3.0 dataset with
`robot_type: so_follower` and 25 FPS. A finalized dataset has this structure:

```text
<dataset>/
├── data/chunk-000/file-000.parquet
├── meta/
│   ├── episodes/chunk-000/file-000.parquet
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos/
    └── observation.images.wrist/chunk-000/file-000.mp4
```

Larger datasets may contain more chunk and file numbers. The required learning
features are:

| Feature | Type and shape | Meaning |
|---|---|---|
| `observation.state` | `float32 (6,)` | measured motor positions |
| `action` | `float32 (6,)` | absolute commanded motor targets |
| `observation.images.wrist` | video `(480, 640, 3)` | wrist RGB camera |

The wrist stream is 25 FPS AV1 video with `yuv420p` pixel format. LeRobot
also adds timestamps plus frame, episode, task, and global row indices.

State and action use this order:

```text
shoulder_pan.pos
shoulder_lift.pos
elbow_flex.pos
wrist_flex.pos
wrist_roll.pos
gripper.pos
```

The five arm values are calibrated degrees. The gripper uses the calibrated
`[0, 100]` range. Values are not normalized. MuJoCo uses radians internally,
but recording converts them to the real-robot convention, including the wrist
roll frame correction.

`action` stores the requested absolute target, not the keyboard delta. If the
measured joint is 180° and a key requests 181°, that frame contains an
observation near 180 and an action of 181. Synthetic trajectories use the same
conversion and feature order.

Before combining simulated and real episodes, verify that both datasets have:

- LeRobot codebase version `v3.0` and 25 FPS;
- the same three feature keys, shapes, names, and camera resolution;
- `robot_type: so_follower`;
- calibrated degrees for the five arm joints and `[0, 100]` for the gripper;
- absolute target actions rather than deltas; and
- the same cube-in-bowl task description and camera semantics.

The validation command checks the recorded schema and decodes the video. The
training wrapper accepts one finalized dataset root, so any real/sim merge must
produce one valid LeRobot v3 root before training.

## Write synthetic trajectories

Participants write ordinary Python in `trajectories/`. Start by copying
[`trajectories/template.py`](trajectories/template.py). The copied template
already imports the participant API; keep its `cube_in_bowl` decorator and
replace the example function body. Each file must define exactly one decorated
trajectory function.

### Participant API

| Call | Purpose |
|---|---|
| `ctx.object_position("name")` | read an object's position after the seeded reset |
| `ctx.current_joints` | read the six measured MuJoCo joint positions in radians |
| `ctx.move_to(position)` | use a tabletop-safe vertical/horizontal path |
| `ctx.move_linear(position)` | follow a straight Cartesian segment |
| `ctx.move_relative(x=..., y=..., z=...)` | move relative to the measured current pose |
| `ctx.move_joints([...])` | move to six absolute MuJoCo joint positions in radians |
| `ctx.open_gripper()` | open to the configured joint limit |
| `ctx.close_gripper(until_contact=True)` | close until the limit or detected contact |
| `ctx.wait(seconds)` | pause the program |
| `ctx.wait_until_settled("name")` | wait for an object to stop moving |
| `ctx.move_home()` | return to the configured home pose |
| `ctx.assert_condition(test, message)` | fail the episode when an assumption is false |

Every movement begins from measured state. Use `ctx.rng` for seeded variation,
so preview and recording reproduce the same randomized path. Mark only intended
contact segments with `allow_contact=True`. The arm has five arm joints, so the
Cartesian planner targets XYZ rather than an arbitrary six-dimensional pose;
use `move_joints()` when a specific joint posture is required.

### Preview and generate

Keep the application running. In a second terminal, preview one unrecorded
episode:

```bash
./scripts/workshop.sh preview participant_trajectory.py 1 13
```

On Windows:

```powershell
.\scripts\workshop.ps1 preview participant_trajectory.py -Seed 13
```

Generate 100 recorded episodes using seeds 1000 through 1099:

```bash
./scripts/workshop.sh generate participant_trajectory.py 100 1000
```

On Windows:

```powershell
.\scripts\workshop.ps1 generate participant_trajectory.py -Episodes 100 -Seed 1000
```

Before recording each seed, the batch runner executes an unrecorded preflight
with that seed. Failed programs are discarded before video encoding. Successful
recordings are saved only when the cube-in-bowl success conditions pass.
Seed-specific planning or execution failures are discarded and the batch
continues with the next seed. Any successful episodes are finalized
automatically when the batch completes, is cancelled, or later encounters an
unrecoverable error.

## Train an ACT policy

Validate the finalized dataset first. Then train ACT for a selected number of
steps on macOS or Linux:

```bash
./scripts/workshop.sh check <dataset-directory>
./scripts/workshop.sh train <dataset-directory> 100000
```

The step count defaults to 100,000 when omitted. To use an NVIDIA GPU on Linux,
install NVIDIA Container Toolkit and give Docker GPU access, then run:

```bash
ROSPIN_GPU=1 ./scripts/workshop.sh train <dataset-directory> 100000
```

On Windows, select the step count and GPU explicitly:

```powershell
.\scripts\workshop.ps1 check <dataset-directory>
.\scripts\workshop.ps1 train <dataset-directory> -Steps 100000
```

Training uses the local dataset, ACT, the PyAV video backend, batch size 8, and
two data workers. Hugging Face upload and Weights & Biases logging are disabled.
Checkpoints and training artifacts are written to:

```text
data/outputs/act_<dataset-directory>/
```

The Windows wrapper can use an NVIDIA GPU when Docker already has GPU access:

```powershell
.\scripts\workshop.ps1 train <dataset-directory> -Gpu -Steps 100000
```

CPU training works on all supported platforms but is substantially slower.
Use consistent successful demonstrations, cover the intended object spawn
range, and keep validation episodes separate when comparing policies.

## Deploy an ACT policy in simulation

A deployable checkpoint is the `pretrained_model` directory inside a training
checkpoint. Pass its path relative to `data/outputs`, followed by the number of
evaluation episodes and the first reset seed:

```bash
ROSPIN_GPU=1 ./scripts/workshop.sh deploy \
  act_<dataset-directory>/checkpoints/010000/pretrained_model \
  10 \
  20000
```

The command starts or recreates the workshop server with the selected CPU or
CUDA image, loads the ACT model and its saved normalization processors, and
runs it in the live MuJoCo task. Open <http://localhost:8000> while it runs to
watch the wrist and perspective cameras. Each rollout receives a fresh seed,
stops on the configured task success condition or episode timeout, and reports
the final success/timeout counts. Press Ctrl+C to request a clean stop.

The policy receives `observation.images.wrist` and the six measured joint
positions at 25 Hz. Simulation radians are converted to the same calibrated
SO-101 units used in the dataset, and the policy's six absolute targets are
converted back to simulation radians and rate-limited by the same controller
used to record the demonstrations. Keyboard and trajectory control are disabled
during a rollout. Deployment can also be started and stopped from the **ACT
policy deployment** panel in the browser.

On Windows:

```powershell
.\scripts\workshop.ps1 deploy `
  act_<dataset-directory>/checkpoints/010000/pretrained_model `
  -Gpu -Episodes 10 -Seed 20000
```
