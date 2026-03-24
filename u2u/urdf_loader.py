# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import json
import os
import pathlib as pl
import shutil as sh

import numpy as np
from loguru import logger
from scipy.spatial.transform import Rotation as R
from urdf_parser_py.urdf import URDF, Joint, Link

from u2u import AssetDir


class UrdfLoader:
    class ParentLinkInfo:
        def __init__(self, parent_link_name: str, joint_name: str):
            self.parent_link_name = parent_link_name
            self.joint_name = joint_name

    class ChildLinkInfo:
        def __init__(self):
            self.child_link_names: list[str] = []

    class MeshLinkInfo:
        def __init__(self, name: str):
            self.name = name
            self.transform: np.ndarray = np.eye(4)

            self.visual_mesh_directories: list[str] = []
            self.visual_mesh_transforms: list[np.ndarray] = []

            self.collision_mesh_directories: list[str] = []
            self.collision_mesh_transforms: list[np.ndarray] = []
            pass

    class LinkInfo:
        def __init__(self, name: str, has_mesh: bool = False):
            self.name = name
            self.has_mesh = has_mesh
            self.transform: np.ndarray = np.eye(4)

    class RevoluteJointInfo:
        def __init__(self, name: str, parent_link_name: str, child_link_name: str):
            self.name = name

            self.local_point_1 = None
            self.local_axis = None
            self.local_point_0 = np.array([0, 0, 0])

            self.global_point_1 = None
            self.global_point_0 = None

            self.parent_link_name = parent_link_name
            self.child_link_name = child_link_name

            self.parent_mesh_link_name = None
            self.child_mesh_link_name = None

            # NOTE: global_axis = child_link.transform @ local_axis
            # NOTE: global_point = child_link.transform @ local_point

    def __init__(self, urdf_path: str):
        self._robot = URDF.from_xml_file(urdf_path)
        self._link_map: dict[str, Link] = self._robot.link_map
        self._joint_map: dict[str, Joint] = self._robot.joint_map
        self._parent_link_map: dict[str, UrdfLoader.ParentLinkInfo] = {}
        self._child_link_map: dict[str, UrdfLoader.ChildLinkInfo] = {}
        self._link_info_map: dict[str, UrdfLoader.LinkInfo] = {}
        self._mesh_link_info_map: dict[str, UrdfLoader.MeshLinkInfo] = {}
        self._revolute_joint_info_map: dict[str, UrdfLoader.RevoluteJointInfo] = {}
        self._root_mesh_link_name: str = None
        self._mesh_link_2_parent_mesh_link_map: dict[str, UrdfLoader.MeshLinkInfo] = {}
        self.urdf_folder = pl.Path(urdf_path).parent.resolve()
        self.package_path: str = None
        self._process()

    # recursively compute the transform of a link based on its parent joint and origin
    def _compute_link_transform(self, link: Link, currentTrans: np.ndarray = np.eye(4, dtype=np.float64)) -> np.ndarray:
        parent_info = self._parent_link_map.get(link.name, None)
        if parent_info is None:
            return currentTrans
        joint = self._joint_map.get(parent_info.joint_name, None)
        if joint is None:
            return currentTrans
        if joint.origin is not None:
            rpy = joint.origin.rpy if joint.origin.rpy is not None else [0, 0, 0]
            xyz = joint.origin.xyz if joint.origin.xyz is not None else [0, 0, 0]
            from scipy.spatial.transform import Rotation as R

            rot = R.from_euler("xyz", rpy).as_matrix()
            trans = np.eye(4)
            trans[:3, 3] = xyz
            trans[:3, :3] = rot
            parent_link = self._link_map.get(parent_info.parent_link_name, None)
            assert parent_link is not None, (
                f"Parent link '{parent_info.parent_link_name}' not found for joint '{joint.name}'"
            )
            currentTrans = self._compute_link_transform(parent_link, trans) @ currentTrans
        return currentTrans

    def _process(self):
        # 1) collect info from joints and links
        self._collect_info()

        # 2) setup link info map and mesh link info map
        self._set_link_basic_info()

        # 3) setup revolute joint local axis and global axis
        self._set_revolute_joint_info()

        # 4) process mesh link info
        self._process_mesh_link_info()

        # 5) find the root mesh link
        self._root_mesh_link_name = self._find_root_mesh_link()
        logger.info(f"Root mesh link: {self._root_mesh_link_name}")

    def _collect_info(self):
        for joint_name, joint in self._joint_map.items():
            # 1) collect parent link info
            self._parent_link_map[joint.child] = UrdfLoader.ParentLinkInfo(joint.parent, joint.name)
            # 2) collect child link info
            if joint.parent not in self._child_link_map:
                self._child_link_map[joint.parent] = UrdfLoader.ChildLinkInfo()
            self._child_link_map[joint.parent].child_link_names.append(joint.child)
            # 3) collect revolute joint info
            if joint.type == "revolute":
                # global_axis = np.array(joint.axis) if joint.axis else np.array([0, 0, 1])
                # global_axis = global_axis / np.linalg.norm(global_axis)
                revolute_joint_info = UrdfLoader.RevoluteJointInfo(
                    name=joint.name,
                    parent_link_name=joint.parent,
                    child_link_name=joint.child,
                )
                self._revolute_joint_info_map[joint.name] = revolute_joint_info

    def _set_link_basic_info(self):
        for link_name, link in self._link_map.items():
            has_mesh = link.collision and link.collision.geometry
            link_info = UrdfLoader.LinkInfo(link_name, has_mesh)
            trans = self._compute_link_transform(link)
            link_info.transform = trans
            self._link_info_map[link.name] = link_info
            if has_mesh:
                mesh_link_info = UrdfLoader.MeshLinkInfo(link_name)
                mesh_link_info.transform = trans
                self._mesh_link_info_map[link_name] = mesh_link_info

    def _set_revolute_joint_info(self):
        for joint_name, joint_info in self._revolute_joint_info_map.items():
            parent_link_info = self._link_info_map.get(joint_info.parent_link_name, None)
            child_link_info = self._link_info_map.get(joint_info.child_link_name, None)

            joint = self._joint_map.get(joint_name, None)
            if parent_link_info is None or child_link_info is None:
                logger.warning(f"Joint '{joint_name}' has missing link info.")
                continue

            # try find the parent mesh link name
            joint_info.parent_mesh_link_name = self._try_get_mesh_parent(parent_link_info.name)
            joint_info.child_mesh_link_name = self._try_get_mesh_child(child_link_info.name)

            # local axis is in the child link's local frame
            joint_info.local_axis = np.array(joint.axis) if joint.axis else np.array([0, 0, 1])
            joint_info.local_axis = joint_info.local_axis / np.linalg.norm(joint_info.local_axis)
            joint_info.local_point_1 = joint_info.local_point_0 + joint_info.local_axis

            # global axis is in the world frame
            joint_info.global_point_1 = child_link_info.transform @ [
                joint_info.local_point_1[0],
                joint_info.local_point_1[1],
                joint_info.local_point_1[2],
                1,
            ]
            joint_info.global_point_0 = child_link_info.transform @ [
                joint_info.local_point_0[0],
                joint_info.local_point_0[1],
                joint_info.local_point_0[2],
                1,
            ]
            joint_info.global_point_1 = joint_info.global_point_1[:3]
            joint_info.global_point_0 = joint_info.global_point_0[:3]

            logger.debug(
                f"Joint '{joint_name}': local axis = {joint_info.local_axis}, "
                f"global axis = {joint_info.global_point_1 - joint_info.global_point_0}, "
                f"global point = {joint_info.global_point_0}, "
                f"local point = {joint_info.local_point_0}"
            )

    def _try_get_mesh_parent(self, link_name: str) -> str:
        # find the first parent link that has a mesh
        mesh_parent_link_name = link_name
        while mesh_parent_link_name is not None:
            if mesh_parent_link_name in self._mesh_link_info_map:
                return mesh_parent_link_name
            info = self._parent_link_map.get(mesh_parent_link_name, None)
            mesh_parent_link_name = info.parent_link_name
        return None

    def _try_get_mesh_child(self, link_name: str) -> str:
        # find first child link that has a mesh
        mesh_child_link_name = link_name
        while mesh_child_link_name is not None:
            if mesh_child_link_name in self._mesh_link_info_map:
                return mesh_child_link_name
            info = self._child_link_map.get(mesh_child_link_name, None)
            if info is None or len(info.child_link_names) == 0:
                return None
            if len(info.child_link_names) > 1:
                logger.warning(
                    f"Joint '{link_name}' has multiple child links: {info.child_link_names}. Using the first one."
                )
            mesh_child_link_name = info.child_link_names[0]
        return None

    def _find_root_mesh_link(self):
        # find the root mesh link, which is the one without any parent
        for link_name, mesh_link_info in self._mesh_link_info_map.items():
            mesh_parent_info = self._mesh_link_2_parent_mesh_link_map.get(link_name, None)
            if mesh_parent_info is None:
                return link_name
        return None

    @staticmethod
    def _xyz_rpy_to_transform(xyz, rpy):
        rot = R.from_euler("xyz", rpy).as_matrix()
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = xyz
        return transform

    def _process_mesh_link_info(self):
        for link_name, mesh_link_info in self._mesh_link_info_map.items():
            link = self._link_map.get(link_name, None)
            if link is None:
                logger.warning(f"Link '{link_name}' not found in URDF.")
                continue

            # Process visual meshes
            if link.visuals:
                for visual in link.visuals:
                    if visual.geometry and visual.geometry.filename:
                        mesh_link_info.visual_mesh_directories.append(visual.geometry.filename)
                        trans = UrdfLoader._xyz_rpy_to_transform(
                            visual.origin.xyz if visual.origin and visual.origin.xyz else [0, 0, 0],
                            visual.origin.rpy if visual.origin and visual.origin.rpy else [0, 0, 0],
                        )
                        mesh_link_info.visual_mesh_transforms.append(trans)

            # Process collision meshes
            if link.collisions:
                for collision in link.collisions:
                    if collision.geometry and collision.geometry.filename:
                        mesh_link_info.collision_mesh_directories.append(collision.geometry.filename)
                        trans = UrdfLoader._xyz_rpy_to_transform(
                            collision.origin.xyz if collision.origin and collision.origin.xyz else [0, 0, 0],
                            collision.origin.rpy if collision.origin and collision.origin.rpy else [0, 0, 0],
                        )
                        mesh_link_info.collision_mesh_transforms.append(trans)

        # setup _mesh_link_parent_map
        for joint_name, joint_info in self._revolute_joint_info_map.items():
            self._mesh_link_2_parent_mesh_link_map[joint_info.child_mesh_link_name] = self._mesh_link_info_map.get(
                joint_info.parent_mesh_link_name, None
            )
        pass

    @property
    def mesh_link_infos(self):
        return self._mesh_link_info_map

    @property
    def revolute_joint_infos(self):
        return self._revolute_joint_info_map

    @property
    def root_mesh_link_name(self):
        return self._root_mesh_link_name

    def _resolve_mesh_path(self, mesh_path: str) -> str:
        mesh_path_pl = pl.Path(mesh_path)
        # package path
        if mesh_path.startswith("package://"):
            mesh_path = mesh_path.replace("package://", "")
            if self.package_path is not None:
                return f"{self.package_path}/{mesh_path}"
            else:
                return str(pl.Path(self.urdf_folder, mesh_path).resolve())

        # absolute path
        if mesh_path_pl.is_absolute():
            return str(mesh_path_pl.resolve())

        # relative path
        return str(pl.Path(self.urdf_folder, mesh_path).resolve())

    def collect_collision_meshes_to_folder(self, folder_path: str) -> dict[str, str]:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mesh_map: dict[str, str] = {}
        mesh_map_inv: dict[str, str] = {}

        for link_name, mesh_link_info in self._mesh_link_info_map.items():
            for i, collision_mesh_path in enumerate(mesh_link_info.collision_mesh_directories):
                collision_mesh_path = self._resolve_mesh_path(collision_mesh_path)
                collision_mesh_path = pl.Path(collision_mesh_path)
                # get the file name (the last part of the path)
                collision_mesh_filename = collision_mesh_path.name
                if collision_mesh_path.exists():
                    new_path = pl.Path(folder_path) / f"{collision_mesh_filename}"
                    mesh_map[str(collision_mesh_path)] = str(new_path)
                    mesh_map_inv[str(new_path)] = str(collision_mesh_path)

                    if new_path.exists():
                        logger.warning(f"Mesh {new_path} already exists, skipping copy.")
                        continue

                    logger.info(f"Copying collision mesh from {collision_mesh_path} to {new_path}")
                    sh.copy(collision_mesh_path, new_path)

        # write the mesh map to a json file
        mesh_map_file = pl.Path(folder_path) / "mesh_map.json"
        with open(mesh_map_file, "w") as f:
            json.dump(mesh_map, f, indent=4)
        logger.info(f"Mesh map saved to {mesh_map_file}")
        # write the inverse mesh map to a json file
        mesh_map_inv_file = pl.Path(folder_path) / "mesh_map_inv.json"
        with open(mesh_map_inv_file, "w") as f:
            json.dump(mesh_map_inv, f, indent=4)
        logger.info(f"Inverse mesh map saved to {mesh_map_inv_file}")

        return mesh_map

    def replace_collision_meshes(self, mesh_map, copy_meshes: bool = False):
        if not copy_meshes:
            logger.info("Not copying meshes, only updating paths in the URDF loader.")

        for link_name, mesh_link_info in self._mesh_link_info_map.items():
            for i, collision_mesh_path in enumerate(mesh_link_info.collision_mesh_directories):
                collision_mesh_path = self._resolve_mesh_path(collision_mesh_path)
                if collision_mesh_path in mesh_map:
                    new_path = mesh_map.get(collision_mesh_path, None)
                    if new_path is not None:
                        if not os.path.exists(new_path):
                            logger.warning(f"Mesh {new_path} does not exist, skipping replacement.")
                            continue
                        mesh_link_info.collision_mesh_directories[i] = new_path
                        logger.info(f"Replacing collision mesh {collision_mesh_path} with {new_path}")
                        if copy_meshes:  # copy the new mesh to the old path
                            sh.copy(new_path, collision_mesh_path)


if __name__ == "__main__":
    urdf_path = AssetDir.urdf_path()
    output_path = AssetDir.output_path(__file__)
    urdf_file = f"{urdf_path}/orca_hand/orcahand_left_extended.urdf"
    loader = UrdfLoader(urdf_file)
    print("Mesh Link Info:")
    for name, info in loader._mesh_link_info_map.items():
        print(f"  {name}: transform = {info.transform}")

    print("Revolute Joint Info:")
    for name, info in loader._revolute_joint_info_map.items():
        print(
            f"  {name}: local axis = {info.local_point_1}, global axis = {info.global_point_1}, global point = {info.global_point_0}, \n"
            f"          parent mesh link = {info.parent_mesh_link_name}, child mesh link = {info.child_mesh_link_name}"
        )
    loader.collect_collision_meshes_to_folder(f"{output_path}/collision_meshes")
    loader.replace_collision_meshes(f"{output_path}/collision_meshes", copy_meshes=False)

    print("Revolute Joint Info:")
    for name, info in loader._revolute_joint_info_map.items():
        print(
            f"  {name}: local axis = {info.local_point_1}, global axis = {info.global_point_1}, global point = {info.global_point_0}, \n"
            f"          parent mesh link = {info.parent_mesh_link_name}, child mesh link = {info.child_mesh_link_name}"
        )
