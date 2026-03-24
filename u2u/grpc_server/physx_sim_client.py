# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
"""
PhysxSimClient gRPC Implementation

This module implements a gRPC client for the PhysxSimService as defined in the
physx_sim_server.proto file.
"""

import argparse
import os
import uuid
from typing import Iterator

import grpc
import numpy as np
from google.protobuf import text_format
from loguru import logger
from pxr import Gf, Usd, UsdGeom, Vt
from scipy.spatial.transform import Rotation as R

from u2u import AssetDir, save_usd

# Import generated protobuf modules
from u2u.proto.api.services.physx.physx_sim_server_pb2 import (
    CreateAndPopulateSceneRequest,
    CreateAndPopulateSceneResponse,
    InitWorldRequest,
    SyncSceneRequest,
    SyncSceneResponse,
)
from u2u.proto.api.services.physx.physx_sim_server_pb2_grpc import (
    PhysxSimServiceStub,
)
from u2u.proto.api.services.physx.scene_config_pb2 import (
    SceneConfig,
)
from u2u.usd_utils import (
    read_usd,
    set_or_add_orient_translate_with_time,
)


class PhysxSimClient:
    """Client for the PhysxSimService gRPC service."""

    def __init__(self, server_address: str = "localhost:50051"):
        """
        Initialize the PhysxSimClient.

        Args:
            server_address: The address of the gRPC server in the format "host:port".
        """
        self.channel = grpc.insecure_channel(server_address)
        self.stub = PhysxSimServiceStub(self.channel)
        logger.info(f"Connected to PhysxSimService at {server_address}")

    def close(self):
        """Close the gRPC channel."""
        self.channel.close()
        logger.info("Closed connection to PhysxSimService")

    def create_and_populate_scene(self, scene_id: str, scene_config: SceneConfig, usd_file_path: str) -> bool:
        """
        Creates and populates a virtual scene in the system based on the specified
        scene ID, configuration, and USD file. The method uploads the USD file in
        chunks and streams the scene creation request to the server. It validates the
        existence of the USD file before processing and ensures orderly upload using
        configurable chunk sizes.

        :param scene_id: Unique identifier for the scene to be created and populated
        :type scene_id: str
        :param scene_config: Configuration details for the scene, including settings and parameters
        :type scene_config: SceneConfig
        :param usd_file_path: The path to the USD file to be uploaded and used for scene creation
        :type usd_file_path: str
        :return: True if the scene is successfully created and populated, False otherwise
        :rtype: bool
        """
        if not os.path.exists(usd_file_path):
            logger.error(f"USD file not found: {usd_file_path}")
            return False

        # Create a generator for the request stream
        def request_generator():
            # First, send the scene configuration
            first_config = CreateAndPopulateSceneRequest.InitScene(scene_id=scene_id, scene_config=scene_config)
            yield CreateAndPopulateSceneRequest(first_config=first_config)

            # Then, send the USD file in chunks
            chunk_size = 4 * 1024 * 1024  # 4MB chunks
            with open(usd_file_path, "rb") as f:
                chunk_index = 0
                while True:
                    data = f.read(chunk_size)
                    if not data:
                        break

                    # Check if this is the last chunk
                    is_last = len(data) < chunk_size or f.tell() == os.path.getsize(usd_file_path)

                    # Create a USD chunk request
                    usd_chunk = CreateAndPopulateSceneRequest.UsdChunk(
                        scene_id=scene_id,
                        data=data,
                        chunk_index=chunk_index,
                        is_last=is_last,
                    )
                    yield CreateAndPopulateSceneRequest(usd_chunk=usd_chunk)

                    chunk_index += 1
                    logger.info(f"Sent chunk {chunk_index} for scene {scene_id}")

                    if is_last:
                        logger.info(f"Completed USD file upload for scene {scene_id}")
                        break

        try:
            # Send the request stream and get the response
            response = self.stub.CreateAndPopulateScene(request_generator())
            success = response.code == CreateAndPopulateSceneResponse.Status.Success
            logger.info(f"Create and populate scene response message: {success}, {response.message}")
            return success
        except grpc.RpcError as e:
            logger.error(f"RPC error in create_and_populate_scene: {e}")
            return False

    def init_world(self, scene_id: str) -> bool:
        """
        Initializes the world using the specified scene ID. This method sends a request
        to initialize the world and waits for the response. If the initialization is
        successful, it returns True; otherwise, logs the error and returns False.

        :param scene_id: Unique identifier of the scene to initialize
        :type scene_id: str
        :return: True if the world is initialized successfully, False otherwise
        :rtype: bool
        """
        request = InitWorldRequest(scene_id=scene_id)
        try:
            response = self.stub.InitWorld(request)
            return response.success
        except grpc.RpcError as e:
            logger.error(f"RPC error in init_world: {e}")
            return False

    def sync_advance(self) -> SyncSceneResponse:
        """
        Advance the simulation by one step and return a single SyncSceneResponse.
        """
        request = SyncSceneRequest(advance_request=SyncSceneRequest.AdvanceRequest())
        return self.stub.SyncScene(request)

    def sync_pause(self) -> SyncSceneResponse:
        """
        Pause the simulation and return a single SyncSceneResponse (may be empty frame).
        """
        request = SyncSceneRequest(pause_request=SyncSceneRequest.PauseRequest())
        return self.stub.SyncScene(request)

    def sync_finish(self) -> SyncSceneResponse:
        """
        Finish the simulation and return a single SyncSceneResponse (may be empty frame).
        """
        request = SyncSceneRequest(finish_request=SyncSceneRequest.FinishRequest())
        return self.stub.SyncScene(request)

    def sync_reset(self) -> SyncSceneResponse:
        """
        Reset the simulation and return a single SyncSceneResponse (may be empty frame).
        """
        request = SyncSceneRequest(reset_request=SyncSceneRequest.ResetRequest())
        return self.stub.SyncScene(request)

    def sync_loop(self, steps: int | None = None, callback=None) -> Iterator[SyncSceneResponse]:
        """
        Repeatedly call SyncScene with AdvanceRequest and yield each response.
        If a callback is provided, it will be invoked with each SyncSceneResponse.

        Args:
            steps: number of steps to advance; if None, loop indefinitely.
            callback: optional callable taking SyncSceneResponse.
        Yields:
            SyncSceneResponse for each step.
        """
        step = 0
        try:
            while steps is None or step < steps:
                response = self.sync_advance()
                if callback:
                    callback(response)
                yield response
                step += 1
        except grpc.RpcError as e:
            logger.error(f"RPC error during sync_loop: {e}")
            raise e

    # Backward-compatible aliases (no-op parameters kept for legacy callers)
    def sync_scene(
        self, scene_id: str | None = None, is_run: bool | None = None, callback=None
    ) -> Iterator[SyncSceneResponse]:
        """Legacy wrapper that ignores scene_id and is_run and uses sync_loop instead."""
        return self.sync_loop(callback=callback)

    def sync_scene_with_callback(self, scene_id: str | None, is_run: bool | None, callback) -> None:
        """Legacy wrapper that ignores scene_id and is_run and advances indefinitely with callback."""
        for _ in self.sync_loop(callback=callback):
            pass


class Client:
    def __init__(self, args):
        self.args = args
        self.client = PhysxSimClient(args.server)
        self.usd_file = args.usd_file
        self.stage = read_usd(self.usd_file)

    def run(self, scene_id: str):
        try:
            # Generate a unique scene ID
            scene_id = str(uuid.uuid4())
            logger.info(f"Using scene ID: {scene_id}")

            # Create a scene configuration
            scene_config = SceneConfig()
            with open(os.path.join(AssetDir.config_path(), "scene_config.proto.txt"), "r") as f:
                text_format.Parse(f.read(), scene_config)

            # Create and populate the scene
            if not self.client.create_and_populate_scene(scene_id, scene_config, self.usd_file):
                logger.error("Failed to create and populate scene")
                return

            # Initialize the world
            if not self.client.init_world(scene_id):
                logger.error("Failed to initialize world")
                return

            # Continuously advance the world, writing frames to USD as they arrive
            try:
                for _ in self.client.sync_loop(
                    steps=200, callback=lambda res, stage=self.stage: write_animation_to_usd(res, stage)
                ):
                    logger.info("Received frame")
            except KeyboardInterrupt:
                logger.info("Interrupted by user; finishing simulation.")
                self.client.sync_finish()

        finally:
            # Close the client
            save_usd(self.stage, f"{AssetDir.output_path(__file__)}/{scene_id}.usd")
            self.client.close()


def pose_to_matrix(pose):
    """Convert Pose message to transformation matrix.

    Args:
        pose: Pose protobuf message containing position and rotation

    Returns:
        4x4 transformation matrix as numpy array
    """

    # Extract position and rotation components
    pos = np.array([pose.translation.x, pose.translation.y, pose.translation.z])
    rot = np.array([pose.rotation.x, pose.rotation.y, pose.rotation.z, pose.rotation.w])

    # Convert quaternion to rotation matrix
    rot_matrix = R.from_quat(rot).as_matrix()

    # Construct 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = pos

    return transform


def write_animation_to_usd(res: SyncSceneResponse, stage: Usd.Stage):
    frame = res.response_frame

    # Process all rigid body poses in this frame
    for rigid in frame.rigid_body_pose:
        prim_path = rigid.prim_path
        timestamp = rigid.timestamp
        transform = rigid.transform
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            logger.warning(f"Prim {prim_path} not found in stage. Defining an Xform at this path.")
            prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        xformable = UsdGeom.Xformable(prim)
        # Author the full transform op at the given time using orientation and translation
        quat = [transform.rotation.w, transform.rotation.x, transform.rotation.y, transform.rotation.z]
        translation = [transform.translation.x, transform.translation.y, transform.translation.z]
        gf_quatf = Gf.Quatf(*quat)
        set_or_add_orient_translate_with_time(xformable, gf_quatf, Gf.Vec3f(*translation), timestamp)
        logger.debug(
            f"Received rigid body pose for {prim_path} at time {timestamp}, gf_quatf: {gf_quatf}, translation: {translation}"
        )

    # Process all soft body point updates in this frame
    for soft_body in frame.soft_body_points:
        prim = stage.GetPrimAtPath(soft_body.prim_path)
        points = [Gf.Vec3f(v.x, v.y, v.z) for v in soft_body.points]
        logger.debug(f"Received soft body points for {soft_body.prim_path} at time {soft_body.timestamp}")
        vec3f_array = Vt.Vec3fArray(points)
        prim.GetAttribute("points").Set(vec3f_array, soft_body.timestamp)


def main():
    """Example usage of the PhysxSimClient."""

    parser = argparse.ArgumentParser(description="PhysxSimClient example")
    parser.add_argument("--server", default="localhost:50051", help="Server address in the format host:port")
    parser.add_argument(
        "--usd-file",
        default="/home/ps/Projects/rbs-physics/assets/usd/HelloWorld.usd",
        help="Path to the USD file to upload",
    )
    args = parser.parse_args()

    client = Client(args)
    scene_id = str(uuid.uuid4())
    client.run(scene_id)


if __name__ == "__main__":
    main()
