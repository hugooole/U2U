# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""
PhysxSimService gRPC Implementation

This module implements a gRPC service for physics simulation as defined in the
physx_sim_server.proto file.
"""

import concurrent.futures
import os

# Add the proto directory to the Python path
from typing import Iterator

import grpc
from google.protobuf import json_format
from loguru import logger
from pxr import Usd
from uipc import Engine, Logger, World

from u2u import AssetDir, read_usd

# Import generated protobuf modules
from u2u.proto.api.services.physx.physx_sim_server_pb2 import (
    CreateAndPopulateSceneRequest,
    CreateAndPopulateSceneResponse,
    InitWorldRequest,
    InitWorldResponse,
    SyncSceneRequest,
    SyncSceneResponse,
)
from u2u.proto.api.services.physx.physx_sim_server_pb2_grpc import (
    PhysxSimServiceServicer,
    add_PhysxSimServiceServicer_to_server,
)
from u2u.scene import Scene
from u2u.usd_parser import UsdParser


class SimulationServer:
    def __init__(self):
        self.scene: Scene | None = None
        self.engine: Engine | None = None
        self.world: World | None = None
        self.stage: Usd.Stage | None = None
        self.scene_config: dict | None = None
        self.scene_id: str | None = None

    def init_scene(self, scene_id: str, scene_config: dict):
        self.scene_id = scene_id
        self.scene_config = scene_config
        self.scene = Scene(self.scene_config)

    def populate_scene(self, usd_file_path: str):
        self.stage = read_usd(usd_file_path)
        parser = UsdParser(self.scene, self.stage)
        self.scene = parser.parse_and_build_scene()

    def init_world(self):
        assert self.scene_id is not None, "scene_id is None"
        assert self.scene is not None, "scene is None"

        # Build a valid workspace directory for the engine using AssetDir utilities
        workdir = os.path.join(AssetDir.output_path(__file__), self.scene_id)
        os.makedirs(workdir, exist_ok=True)

        self.engine = Engine(backend_name="cuda", workspace=workdir)
        self.world = World(self.engine)
        self.world.init(self.scene)


class PhysxSimServiceImpl(PhysxSimServiceServicer):
    """Implementation of the PhysxSimService gRPC service."""

    def __init__(self, _server: SimulationServer):
        # set cuda backend log level to warn
        Logger.set_level(Logger.Warn)
        self._server = _server
        # Directory where uploaded USD files will be stored
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.uploads_dir = os.path.join(project_root, "uploads")
        os.makedirs(self.uploads_dir, exist_ok=True)

    def CreateAndPopulateScene(
        self, request_iterator: Iterator[CreateAndPopulateSceneRequest], context
    ) -> CreateAndPopulateSceneResponse:
        """
        Create a scene with SceneConfig and populate it with a USD file.

        This method handles streaming uploads of USD files in chunks.
        """
        scene_id = None
        temp_file_path = None
        chunk_count = 0

        try:
            for request in request_iterator:
                if request.HasField("first_config"):
                    # Handle the initial scene configuration
                    scene_id = request.first_config.scene_id
                    logger.info(f"Received scene configuration for scene {scene_id}")

                    # Create a new scene with the provided configuration
                    scene_config = request.first_config.scene_config
                    config_dict = json_format.MessageToDict(
                        scene_config,
                        always_print_fields_with_no_presence=True,
                        preserving_proto_field_name=True,
                    )
                    config_dict.update(
                        {
                            "gravity": [[scene_config.gravity.x], [scene_config.gravity.y], [scene_config.gravity.z]],
                        }
                    )

                    logger.info(f"Scene configuration:\n{config_dict}")

                    self._server.init_scene(scene_id, config_dict)
                    logger.info(f"Created scene {scene_id} with configuration")

                elif request.HasField("usd_chunk"):
                    # Handle USD file chunks
                    chunk = request.usd_chunk

                    # Verify that a scene has been initialized
                    if scene_id is None:
                        return CreateAndPopulateSceneResponse(
                            message="Received USD chunk before scene initialization",
                            code=CreateAndPopulateSceneResponse.Status.Failed,
                        )

                    # Check that the chunk belongs to the correct scene
                    if chunk.scene_id != scene_id:
                        return CreateAndPopulateSceneResponse(
                            message=f"Chunk scene ID {chunk.scene_id} does not match any scene ID in current scene dict",
                            code=CreateAndPopulateSceneResponse.Status.Failed,
                        )

                    # Create a temporary file for the first chunk
                    if temp_file_path is None:
                        temp_file_name = f"{scene_id}.usd"
                        temp_file_path = os.path.join(self.uploads_dir, temp_file_name)
                        logger.info(f"Creating temporary file {temp_file_path}")

                    # Write the chunk data to the temporary file
                    with open(temp_file_path, "ab") as f:
                        f.write(chunk.data)

                    chunk_count += 1
                    logger.info(f"Received chunk {chunk.chunk_index} for scene {scene_id}")

                    # If this is the last chunk, finalize the file
                    if chunk.is_last:
                        final_file_name = f"{scene_id}.usd"
                        final_file_path = os.path.join(self.uploads_dir, final_file_name)
                        os.rename(temp_file_path, final_file_path)
                        logger.info(f"Completed USD file upload for scene {scene_id} with {chunk_count} chunks")
                        self._server.populate_scene(final_file_path)
                        logger.info(f"Successfully populate scene {scene_id} with usd file {final_file_path}")
                        return CreateAndPopulateSceneResponse(
                            message=f"Successfully created scene and uploaded USD file with {chunk_count} chunks",
                            code=CreateAndPopulateSceneResponse.Status.Success,
                        )

            # If we get here, the upload was incomplete
            return CreateAndPopulateSceneResponse(
                message="Incomplete USD file upload (missing last chunk)",
                code=CreateAndPopulateSceneResponse.Status.Failed,
            )

        except Exception as e:
            logger.error(f"Error in CreateAndPopulateScene: {str(e)}")
            # Clean up temporary file if it exists
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return CreateAndPopulateSceneResponse(
                message=f"Error: {str(e)}",
                code=CreateAndPopulateSceneResponse.Status.Failed,
            )

    def InitWorld(self, request: InitWorldRequest, context) -> InitWorldResponse:
        """
        Initialize a world for the specified scene.

        This method parses the USD file and builds the scene.
        """
        scene_id = request.scene_id

        try:
            self._server.init_world()

            logger.info(f"Initialized world for scene {scene_id}")

            return InitWorldResponse(
                message=f"Successfully initialized world for scene {scene_id}",
                success=True,
            )

        except Exception as e:
            logger.error(f"Error in InitWorld: {str(e)}")
            return InitWorldResponse(
                message=f"Error: {str(e)}",
                success=False,
            )

    def SyncScene(self, request: SyncSceneRequest, context) -> SyncSceneResponse:
        """
        Synchronize the scene state with a single response.

        This method advances the simulation (if requested) and returns a single
        SyncSceneResponse containing a ResponseFrame (rigid body poses and soft body points).
        """
        # Basic checks to avoid attribute errors if world/scene are not initialized
        if self._server.world is None or self._server.scene is None:
            logger.error("World or Scene is not initialized")
            return SyncSceneResponse(response_frame=SyncSceneResponse.ResponseFrame())

        if request.HasField("advance_request"):
            # Step the simulation
            self._server.world.advance()
            self._server.world.retrieve()

            # Build a single response frame that aggregates all updates for this frame
            frame_idx = self._server.world.frame()
            logger.info(f"Sending frame {frame_idx}")

            response_frame: SyncSceneResponse.ResponseFrame = SyncSceneResponse.ResponseFrame()

            for data in self._server.scene.animation_iterator(frame_idx):
                if "transform" in data:
                    rigid_frame = response_frame.rigid_body_pose.add()
                    rigid_frame.prim_path = data["prim_path"]
                    rigid_frame.timestamp = data["timestamp"]
                    rigid_frame.transform.CopyFrom(transform_to_pose(data["transform"]))
                    logger.info(f"Rigid body pose: {rigid_frame}")
                elif "points" in data:
                    soft_frame = response_frame.soft_body_points.add()
                    soft_frame.prim_path = data["prim_path"]
                    soft_frame.timestamp = data["timestamp"]
                    for point in data["points"]:
                        soft_frame.points.append(vec3f_from_numpy(point))
                    logger.info(f"Soft body points: {soft_frame}")

            return SyncSceneResponse(response_frame=response_frame)

        elif request.HasField("pause_request"):
            logger.info("Pause request received")
            # No world change here; return an empty frame or current state if desired
            return SyncSceneResponse(response_frame=SyncSceneResponse.ResponseFrame())

        elif request.HasField("finish_request"):
            logger.info("Finish request received")
            # Implement any cleanup if needed and return an empty frame
            return SyncSceneResponse(response_frame=SyncSceneResponse.ResponseFrame())

        elif request.HasField("reset_request"):
            logger.info("Reset request received, but it not implemented yet.")
            # If you have a reset method, call it; otherwise return an empty frame
            return SyncSceneResponse(response_frame=SyncSceneResponse.ResponseFrame())

        else:
            logger.warning("Unknown SyncSceneRequest payload")
            return SyncSceneResponse(response_frame=SyncSceneResponse.ResponseFrame())


def transform_to_pose(transform):
    """Convert a transformation matrix to a Pose message."""
    from u2u.proto.api.math.pose_pb2 import Pose
    from u2u.proto.api.math.vec_pb2 import Vec3d

    # Extract translation from the transformation matrix
    translation = Vec3d(x=float(transform[0, 3]), y=float(transform[1, 3]), z=float(transform[2, 3]))

    # Extract rotation from the transformation matrix
    # Convert the rotation matrix to quaternion
    from scipy.spatial.transform import Rotation as R

    rotation = R.from_matrix(transform[:3, :3]).as_quat()

    # Create and return a Pose message
    return Pose(translation=translation, rotation=rotation_to_proto(rotation))


def rotation_to_proto(rotation):
    """Convert a rotation quaternion to a Quaternion message."""
    from u2u.proto.api.math.rotation_pb2 import Quaternion

    return Quaternion(w=float(rotation[3]), x=float(rotation[0]), y=float(rotation[1]), z=float(rotation[2]))


def vec3f_from_numpy(point):
    """Convert a numpy array to a Vec3f message."""
    from u2u.proto.api.math.vec_pb2 import Vec3f

    return Vec3f(x=float(point[0]), y=float(point[1]), z=float(point[2]))


def serve(servicer: SimulationServer, port=50051):
    """Start the gRPC server."""
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=10))
    # Pass the SimulationServer instance into the service implementation
    add_PhysxSimServiceServicer_to_server(PhysxSimServiceImpl(servicer), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    logger.info(f"Server started on port {port}")
    return server


if __name__ == "__main__":
    servicer = SimulationServer()
    server = serve(servicer)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(0)
