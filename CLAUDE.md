# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**U2U** is a Python library that bridges Universal Scene Description (USD) and libuipc (physics simulation engine) to simplify physics-based robotics simulation. It provides a comprehensive pipeline for importing USD scenes, configuring physics properties, controlling robots through a task-based system, and exporting animations back to USD.

**Key Technologies:**
- USD (Universal Scene Description) for scene representation
- libuipc for physics simulation with CUDA acceleration
- Polyscope for 3D visualization
- Python 3.11+ (NumPy 1.26.0 pinned)

## Development Commands

**⚠️ CRITICAL: This project uses `uv` as the package manager. Always use `uv run <script>` to execute Python scripts. Do NOT use `python` or `python3` directly - they will fail with "command not found" errors.**

### Environment Setup
```bash
# Install dependencies (uses uv package manager with custom PyPI indices)
uv sync --all-groups

# Install Git LFS (required for large asset files)
git lfs install
git lfs pull
```

### Building
```bash
# Build the package
uv build

# Build outputs to dist/ directory
```

### Code Quality
```bash
# Run pre-commit hooks (ruff linter and formatter)
pre-commit run --all-files

# Ruff is configured with:
# - Line length: 120
# - Auto-fix enabled
# - Format and check in pre-commit
```

### Running Examples
```bash
# IMPORTANT: Always use 'uv run' to execute Python scripts in this project
# DO NOT use 'python' or 'python3' directly - they are not configured in the virtual environment

# Run examples using uv
uv run examples/pick_and_place_with_mplib.py
uv run examples/articulation_demo.py
uv run examples/hello_world.py
uv run examples/cartpole.py
```

**Why `uv run`?**
- This project uses `uv` as the package manager and task runner
- `uv run` automatically activates the virtual environment and uses the correct Python interpreter
- Direct `python` commands will fail with "command not found" errors


### GRPC Proto Generation
When editing proto files:
```bash
# Delete old generated code
find u2u/proto -type f \( -name "*.py" -o -name "*.pyi" \) -delete

# Generate new code from proto files
# Note: This is one of the few cases where 'python -m' is used directly
python -m grpc.tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. $(find u2u/proto -type f -name "*.proto")
```

## Architecture

### Core Components

**Scene (`u2u/scene.py`)**
- Extends `uipc.Scene` as the main simulation container
- Manages contact tabular for friction and collision between objects
- Tracks geometry and robot dictionaries
- Handles transform updates for both rigid bodies and articulated robots
- Supports up-axis configuration (X, Y, or Z)
- Provides bidirectional reference to `World` via weak proxy (`scene.world` property)
- GPU-accelerated state manipulation:
  - `reset_affine_body_state()`: Reset rigid body transforms/velocities using Warp kernels
  - `affine_body_state_accessor`: Lazy-initialized accessor for affine body state
  - `finite_element_state_accessor`: Lazy-initialized accessor for FEM state

**World (`u2u/world.py`)**
- Extends `uipc.World` as a wrapper for physics simulation engine
- Maintains reference to `Scene` instance via `_scene` attribute
- Establishes bidirectional Scene-World reference using weak proxy pattern
- Provides `scene` property to access the managed Scene instance
- Prevents circular references through weak proxy back-reference

**Pipeline (`u2u/pipeline.py`)**
- `PipelineBase`: Abstract base class for all simulations
- Integrates USD stage loading, physics world initialization, and Polyscope visualization
- Provides GUI with joint sliders for robot control
- Manages task queue for sequential/asynchronous operations
- Optional `use_warp` parameter enables Warp GPU acceleration
- Must implement one abstract method:
  - `setup_config()`: Return physics configuration dictionary
- Optional methods (with default empty implementations):
  - `setup_contact_tabular()`: Configure contact properties between objects
  - `user_build_scene()`: Custom scene setup logic
- Customization hooks:
  - `after_world_init()`: Called after World initialization for custom setup
  - `user_define_gui()`: Add custom GUI elements to Polyscope interface

**UsdParser (`u2u/usd_parser.py`)**
- Parses USD stages into physics-ready scenes
- Handles unit conversion (meters, kilograms)
- Automatically sets up gravity based on USD up-axis
- Uses specialized builders for different object types

**Scene Builders (`u2u/scene_builder/`)**
- `ArticulationBuilder`: Handles articulated robots with joints (revolute, prismatic, fixed)
  - Joint constitution apply order: `body0_slot`, `body0_indices`, `body1_slot`, `body1_indices`, `strength`
  - Default constraint strength: 1000.0 (revolute), 100.0 (prismatic)
- `RigidBodyBuilder`: Processes rigid bodies with collision geometry
  - Uses `uipc.unit.MPa` for resistance coefficient (e.g., `kappa=100 * MPa`)
- `ClothBuilder`: Manages cloth/soft body simulations
- `DeformableBuilder`: Handles deformable objects
- Each builder extracts geometry, applies physics properties, and sets up constraints

**Articulation (`u2u/scene_builder/articulation.py`)**
- Represents articulated robots with joint hierarchies
- Supports three joint types: `RevoluteJoint`, `PrismaticJoint`, `FixedJoint`
- Joint control modes: NONE, POSITION, VELOCITY (use `set_joint_effort()` for force/torque control with NONE mode)
- Manages kinematic tree, joint limits, and control parameters
- Handles root pose transformations and forward kinematics
- Joint geometry attributes use string-based lookup:
  - Revolute: `"angle"`, `"aim_angle"`, `"external_torque"`, `"driving/is_constrained"`, `"external_torque/is_constrained"`
  - Prismatic: `"distance"`, `"aim_distance"`, `"external_force"`, `"driving/is_constrained"`, `"external_force/is_constrained"`
- `set_joint_effort()` accepts `np.ndarray` (use `np.array([value])` for scalar)

**Task Queue (`u2u/task_queue.py`)**
- Async task system with priorities and dependencies
- Task states: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
- Supports callbacks on completion/failure
- Use for motion planning, sequential robot operations, etc.

**Controllers (`u2u/controllers.py`)**
- `PDController`: Proportional-Derivative control (`u = Kp*(target-pos) - Kd*vel`)
- `PIDController`: PD + integral term with anti-windup protection
- Both support optional force/torque saturation limits
- Tuning: `Kd = ζ × 2√(Kp × m)` where ζ=1.0-1.2 for slight overdamping

**Pose (`u2u/pose.py`)**
- Robust 3D pose representation with position and orientation (quaternion)
- Supports interpolation, inverse, and composition operations
- Converts to/from transformation matrices and USD Gf types

### Data Flow

1. **USD Import** → UsdParser reads USD stage, extracts prims with physics APIs
2. **Scene Building** → Builders create SimplicialComplex geometries and apply constitutions
3. **World Init** → Physics engine initializes with configured scene
4. **Simulation Loop** → TaskQueue processes tasks, world steps, transforms update USD
5. **Export** → Updated transforms written back to USD for animation playback

### Key Patterns

**Joint State Reset**: `reset_joint_position()` sets `joint_is_constrained=False` for ALL joints. Re-enable constraints explicitly after calling it:
```python
scene.reset_joint_position(robot_name="/robot", joint_positions={...})
# IMPORTANT: re-constrain joints that should hold position
robot.set_joint_constrained("/robot/joint1", True)
robot.set_joint_position("/robot/joint1", np.array([target]))
```

**Contact Configuration**: Always define contact elements and insert into contact_tabular with friction/resistance coefficients:
```python
table_elem = self._contact_tabular.create("table")
cube_elem = self._contact_tabular.create("cube")
self._contact_tabular.insert(table_elem, cube_elem, friction_rate, resistance)
table_elem.apply_to(self.scene.get_geometry("/World/table"))
```

**Robot Control**: Set robot reference, control joints, use tasks for motion:
```python
self.set_robot("/World/robot_name")
# Control mode is automatically set when calling position/velocity/effort methods
self.robot.set_joint_position(joint_name, target_position)  # Auto-sets POSITION mode
# Or use velocity control
self.robot.set_joint_velocity(joint_name, target_velocity)  # Auto-sets VELOCITY mode
# Or use force/torque control
self.robot.set_joint_effort(joint_name, np.array([force_value]))  # Auto-sets NONE mode with force constraint
self.task_queue.add(MoveToTargetTask(...))
```

## Project Structure

```
u2u/
├── scene.py              # Main Scene class extending uipc.Scene
├── pipeline.py           # PipelineBase for simulation workflows
├── usd_parser.py         # USD stage parser
├── usd_utils.py          # USD manipulation utilities
├── task_queue.py         # Async task system
├── pose.py               # 3D pose utilities
├── mesh_factory.py       # Mesh generation and processing
├── urdf_loader.py        # URDF import
├── urdf2usd.py           # URDF to USD conversion
├── scene_builder/        # Object-specific builders
│   ├── articulation.py   # Articulated robots
│   ├── rigid_body.py     # Rigid bodies
│   ├── cloth.py          # Cloth simulation
│   └── deformable_body.py # Deformable objects
├── grpc_server/          # gRPC service for remote control
└── proto/                # Protocol buffer definitions

assets/
├── usd/                  # USD scene files
├── urdf/                 # URDF robot descriptions
├── trimesh/              # Triangle meshes
└── tetmesh/              # Tetrahedral meshes

examples/                 # Example simulations
configs/                  # Scene configuration files
docs/                     # Sphinx documentation
```

## Dependencies

**Required Python Version**: 3.11+ (NumPy 1.26.0 is pinned, do not upgrade)

**Core Dependencies**:
- pyuipc: Physics simulation engine using libuipc
- usd-core: USD library
- polyscope: 3D visualization
- trimesh, tetgen: Mesh processing
- scipy: Scientific computing
- warp-lang: NVIDIA Warp for GPU-accelerated kernels (required for `reset_affine_body_state()`)
- loguru: Structured logging (used in articulation callbacks for debug-level joint state tracing)

**Optional Dependencies**:
- examples group: mplib (motion planning), toppra, tqdm
- dev group: mypy (Linux only), sphinx, pybind11, pytest
