===========
u2u
===========

**u2u** is a Python library that bridges Universal Scene Description (USD) and libuipc (physics simulation engine) to simplify physics-based robotics simulation. It provides a comprehensive pipeline for importing USD scenes, configuring physics properties, controlling robots through a task-based system, and exporting animations back to USD.

For Development
===============

GRPC Service
------------

For edit proto and generate python code

.. code-block:: bash

    # delete old code
    find u2u/proto -type f \( -name "*.py" -o -name "*.pyi" \) -delete

    # for windows powershell
    Get-ChildItem -Path "u2u\proto" -Recurse -Include "*.py","*.pyi" | Remove-Item

    # generate grpc code
    python -m grpc.tools.protoc -I. --python_out=. --pyi_out=. --grpc_python_out=. $(find u2u/proto -type f -name "*.proto")

Features
========

- **USD Integration**: Import and export USD scenes with physics properties
- **Physics Simulation**: Configure and run physics simulations using libuipc
- **Robot Control**: Task-based system for robot manipulation and control
- **Articulation Support**: Handle articulated robots with joint control
- **Contact Modeling**: Configure contact properties between different objects
- **Visualization**: Integration with Polyscope for 3D visualization
- **Pose Manipulation**: Robust pose representation and transformation utilities
- **Multi-Instance Simulation**: Run multiple robot instances simultaneously in a single scene

.. image:: docs/images/multi_instances.jpg
   :alt: Multi-instance simulation visualization
   :width: 600px

*Multi-instance simulation showing 100 robot instances running in parallel with full physics simulation*

Requirements
============

- Python 3.11+
- CUDA-compatible GPU (for GPU acceleration)

Dependencies
============

Core Dependencies:
-----------------

- loguru: Logging utility
- polyscope: 3D visualization
- scipy: Scientific computing
- tetgen: Tetrahedral mesh generation
- trimesh: Mesh processing
- types-usd: Type hints for USD
- pyuipc: Physics simulation engine

Optional Dependencies:
--------------------

For examples:
~~~~~~~~~~~~~

- mplib: Motion planning library
- tqdm: Progress bar

For development:
~~~~~~~~~~~~~~~

- mypy: Static type checking
- pybind11: C++ bindings
- sphinx: Documentation generation

Installation
============

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/roboscience-ai/rbs-physics.git
      cd rbs-physics

2. Install Git LFS (required for large files):

   .. code-block:: bash

      git lfs install
      git lfs pull

3. Install dependencies using uv:

   .. code-block:: bash

      uv sync --all-groups

Usage
=====

Basic Example
------------

Here's a simple example of how to use rbs-physics for a pick-and-place operation:

.. code-block:: python

   import os.path as osp
   from u2u import AssetDir
   from u2u.pipeline import PipelineBase
   from u2u.scene import Scene

   class MySimulation(PipelineBase):
       def __init__(self, workdir, usd_path):
           super().__init__(workdir, usd_path)

       def setup_config(self):
           config = Scene.default_config()
           config["contact"]["enable"] = True
           config["contact"]["friction"]["enable"] = True
           return config

       def setup_contact_tabular(self):
           # Configure contact properties
           self.set_robot("/World/robot")
           # ...

   # Run the simulation
   sim = MySimulation(
       workdir=AssetDir.output_path(__file__),
       usd_path=osp.join(AssetDir.usd_path(), "my_scene.usd")
   )
   sim.run()

Running Examples
---------------

The package includes several example simulations:

1. Pick and Place with Motion Planning:

   .. code-block:: bash

      uv run examples/pick_and_place_with_mplib.py

2. Two Hands Articulation:

   .. code-block:: bash

      uv run examples/two_hands.py

Documentation
============

For more detailed documentation, please refer to the docstrings in the code or generate the documentation using Sphinx:

.. code-block:: bash

   cd docs
   uv run sphinx-build -b html source build

Contributing
============

Contributions are welcome! Please feel free to submit a Pull Request.

License
========

Please refer to the LICENSE file in the repository for licensing information.
