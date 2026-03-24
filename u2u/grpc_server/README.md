# PhysxSimService gRPC Implementation

This module implements a gRPC service for physics simulation as defined in the `physx_sim_server.proto` file.

## Overview

The implementation consists of:

1. Server: Handles physics simulation with three RPC methods matching the current proto:
   - CreateAndPopulateScene: Creates a scene using SceneConfig and uploads a USD file via a streaming request
   - InitWorld: Initializes the physics world for a previously created scene
   - SyncScene: Streams simulation frames (rigid body poses and soft body points) to the client

2. Client: Provides a convenient interface for interacting with the server

## Implementation Details

### Server

The server implementation (`physx_sim_service.py`) provides:

- A `PhysxSimServicer` class that implements all three RPC methods
- Scene management with configuration storage
- File handling for USD uploads using chunked streaming
- Physics world initialization and simulated updates for scene synchronization

### Client

The client implementation (`physx_sim_client.py`) provides:

- A `PhysxSimClient` class for connecting to the server
- Methods for creating and populating a scene, initializing the world, and synchronizing scene state
- An example main that performs an end-to-end flow using command-line options

### Protocol Buffer

The service is defined in `u2u/proto/api/services/physx/physx_sim_server.proto` and includes:

- `PhysxSimService` service with three RPC methods: CreateAndPopulateScene, InitWorld, SyncScene
- Message types for requests and responses

## Usage

### Running the Server

To start the server:

```python
from u2u.grpc_server.physx_sim_service import serve

# Start the server on the default port (50051)
server = serve()

# Keep the server running until interrupted
try:
    server.wait_for_termination()
except KeyboardInterrupt:
    server.stop(0)
```

Or run the server module directly:

```bash
python -m u2u.grpc_server.physx_sim_service
```

### Using the Client

#### Create and Populate a Scene, Then Initialize the World

```python
import os
import uuid
from google.protobuf import text_format
from u2u.grpc_server.physx_sim_client import PhysxSimClient
from u2u.proto.api.services.physx.scene_config_pb2 import SceneConfig
from u2u import AssetDir

client = PhysxSimClient("localhost:50051")
try:
    # Unique scene ID
    scene_id = str(uuid.uuid4())

    # Load SceneConfig from a text proto file (example asset)
    scene_config = SceneConfig()
    with open(os.path.join(AssetDir.config_path(), "scene_config.proto.txt"), "r") as f:
        text_format.Parse(f.read(), scene_config)

    # Upload config and USD file in a single streaming RPC
    usd_file_path = "/path/to/file.usd"
    ok = client.create_and_populate_scene(scene_id, scene_config, usd_file_path)
    if not ok:
        raise RuntimeError("CreateAndPopulateScene failed")

    # Initialize the physics world
    if not client.init_world(scene_id):
        raise RuntimeError("InitWorld failed")
finally:
    client.close()
```

#### Synchronizing Scene State

You can consume streaming frames either with a generator or a callback helper.

Generator-based iteration:

```python
from u2u.grpc_server.physx_sim_client import PhysxSimClient

client = PhysxSimClient("localhost:50051")
try:
    scene_id = "existing-scene-id"
    for response in client.sync_scene(scene_id, True):
        if response is None:
            break  # stream ended due to error
        if response.HasField("response_frame"):
            frame = response.response_frame
            if frame.HasField("rigid_body_pose"):
                pose = frame.rigid_body_pose
                print("Rigid:", pose.prim_path, pose.timestamp)
            elif frame.HasField("soft_body_points"):
                pts = frame.soft_body_points
                print("Soft:", pts.prim_path, len(pts.points), "points")
        # Stop condition example (not shown): break after some frames
finally:
    client.close()
```

Callback helper:

```python
from u2u.grpc_server.physx_sim_client import PhysxSimClient

client = PhysxSimClient("localhost:50051")
try:
    scene_id = "existing-scene-id"

    def on_response(res):
        if res is None:
            print("Stream error or ended")
            return
        if res.HasField("response_frame"):
            frame = res.response_frame
            if frame.HasField("rigid_body_pose"):
                print("Rigid frame received")
            elif frame.HasField("soft_body_points"):
                print("Soft frame received")

    client.sync_scene_with_callback(scene_id, True, on_response)
finally:
    client.close()
```

### Command-line Example

The client module provides a simple end-to-end example you can run:

```bash
python -m u2u.grpc_server.physx_sim_client --server localhost:50051 --usd-file /path/to/file.usd
```

This script:
- Generates a unique scene_id
- Loads a SceneConfig from assets (scene_config.proto.txt)
- Calls CreateAndPopulateScene to upload the USD file and config
- Calls InitWorld
- Starts streaming frames using sync_scene_with_callback

### Running Tests

If you have tests in your environment, run them with:

```bash
python -m u2u.grpc_server.test_physx_sim_service
```

## Dependencies

- grpcio
- grpcio-tools
- protobuf

## References

This implementation uses a chunked upload approach similar to: https://ops.tips/blog/sending-files-via-grpc/
