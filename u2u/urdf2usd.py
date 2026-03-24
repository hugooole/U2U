# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf
import pathlib as pl
import trimesh
from loguru import logger
import numpy as np
from .urdf_loader import UrdfLoader


class Urdf2Usd:
    def __init__(
        self,
        usd_stage: Usd.Stage,
        root_prim: Usd.Prim = None,
        package_path: str = None,
        with_visual_mesh: bool = True,
        with_mesh_subset: bool = True,
    ):
        self.usd_stage = usd_stage
        if root_prim is None:
            self.root_prim = self.usd_stage.DefinePrim("/Robot", "Xform")
        else:
            self.root_prim = root_prim
        self.urdf_loader: UrdfLoader = None
        self.package_path: str = package_path
        self.urdf_folder: str = ""
        self.with_visual_mesh = with_visual_mesh
        self.with_mesh_subset = with_mesh_subset

    @staticmethod
    def _quat_from_veca_to_vecb(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.array(a, dtype=np.float64)
        b = np.array(b, dtype=np.float64)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        dot = np.dot(a, b)
        if np.isclose(dot, 1.0):
            # Same direction
            return np.array([1.0, 0.0, 0.0, 0.0])
        if np.isclose(dot, -1.0):
            # Opposite direction: find arbitrary orthogonal axis
            axis = np.cross(a, [1, 0, 0])
            if np.linalg.norm(axis) < 1e-6:
                axis = np.cross(a, [0, 1, 0])
            axis /= np.linalg.norm(axis)
            return np.array([0.0, *axis])
        # Normal case
        axis = np.cross(a, b)
        axis /= np.linalg.norm(axis)
        angle = np.arccos(dot)
        w = np.cos(angle / 2)
        xyz = axis * np.sin(angle / 2)
        return np.array([w, *xyz])

    def _add_mesh_subset(self, prim: UsdGeom.Mesh, link_name: str):
        prim = UsdGeom.Mesh(prim)
        face_count = len(prim.GetFaceVertexCountsAttr().Get())
        all_indices = list(range(face_count))
        mesh_subset_path = f"{prim.GetPath()}/MeshSubset_{link_name}"
        logger.info(f"Adding MeshSubset at {mesh_subset_path} with {face_count} faces.")
        full_subset = UsdGeom.Subset.Define(self.usd_stage, mesh_subset_path)
        full_subset.CreateIndicesAttr(all_indices)
        full_subset.CreateElementTypeAttr("face")  # 类型为面

    def from_urdf_file(self, urdf_path: str, mesh_map: str = None):
        self.urdf_loader = UrdfLoader(urdf_path)

        if mesh_map is not None:
            self.urdf_loader.replace_collision_meshes(mesh_map, copy_meshes=False)

        self.urdf_folder = pl.Path(urdf_path).parent.resolve()
        logger.info(f"Loading URDF from: {urdf_path}")
        logger.info(f"URDF folder: {self.urdf_folder}")
        if self.package_path is None:
            self.package_path = self.urdf_folder
        logger.info(f"Package path: {self.package_path}")

        mesh_link_infos = self.urdf_loader.mesh_link_infos

        # 1) create links
        root_path = self.root_prim.GetPath()
        # add articulation root API on the root prim
        UsdPhysics.ArticulationRootAPI.Apply(self.root_prim.GetPrim())

        for link_name, info in self.urdf_loader.mesh_link_infos.items():
            link_prim = UsdGeom.Xform.Define(self.usd_stage, f"{root_path}/{link_name}")

            # add rigid body API on the link prim
            UsdPhysics.RigidBodyAPI.Apply(link_prim.GetPrim())
            UsdPhysics.CollisionAPI.Apply(link_prim.GetPrim())

            # set link transform
            link_prim.AddTransformOp().Set(Gf.Matrix4d(info.transform.T))

            # add visual and collision meshes
            if self.with_visual_mesh:
                visuals = UsdGeom.Xform.Define(self.usd_stage, f"{link_prim.GetPath()}/visuals")

                for path, trans in zip(info.visual_mesh_directories, info.visual_mesh_transforms):
                    mesh_dir_pl = pl.Path(path)
                    name = mesh_dir_pl.stem
                    # remove extension in name
                    name = name.split(".")[0]
                    mesh_prim = UsdGeom.Mesh.Define(self.usd_stage, f"{visuals.GetPath()}/{name}")
                    mesh_prim = self._urdf_mesh_to_usd_mesh(mesh_prim, trans, path)
                    if self.with_mesh_subset:
                        self._add_mesh_subset(mesh_prim.GetPrim(), link_name)

            # add collision meshes
            collisions = UsdGeom.Xform.Define(self.usd_stage, f"{link_prim.GetPath()}/collisions")

            for path, trans in zip(info.collision_mesh_directories, info.collision_mesh_transforms):
                mesh_dir_pl = pl.Path(path)
                name = mesh_dir_pl.stem
                # remove extension in name
                name = name.split(".")[0]
                mesh_prim = UsdGeom.Mesh.Define(self.usd_stage, f"{collisions.GetPath()}/{name}")
                # add collision api
                UsdPhysics.CollisionAPI.Apply(mesh_prim.GetPrim())
                mesh_prim = self._urdf_mesh_to_usd_mesh(mesh_prim, trans, path)

        # 2) create revolute joints
        revolute_joint_infos = self.urdf_loader.revolute_joint_infos

        # create a scope for the joint prims
        joint_scope = UsdGeom.Scope.Define(self.usd_stage, f"{root_path}/joints")

        for name, info in revolute_joint_infos.items():
            joint_prim = UsdPhysics.RevoluteJoint.Define(self.usd_stage, f"{joint_scope.GetPath()}/{name}")
            # joint axis is defined as Point + Direction
            axis = info.global_point_1 - info.global_point_0
            axis = axis / np.linalg.norm(axis)  # normalize the axis
            point = info.global_point_0

            body_0_name = info.parent_mesh_link_name
            body_0_link: UrdfLoader.MeshLinkInfo = mesh_link_infos.get(body_0_name, None)
            body_1_name = info.child_mesh_link_name
            body_1_link: UrdfLoader.MeshLinkInfo = mesh_link_infos.get(body_1_name, None)

            trans0: np.ndarray = body_0_link.transform if body_0_link else np.eye(4)
            trans1: np.ndarray = body_1_link.transform if body_1_link else np.eye(4)

            inv_trans0 = np.linalg.inv(trans0)
            inv_trans1 = np.linalg.inv(trans1)

            localPos0 = inv_trans0 @ [point[0], point[1], point[2], 1]
            localPos0 = localPos0[:3]  # Convert to 3D vector
            localPos1 = inv_trans1 @ [point[0], point[1], point[2], 1]
            localPos1 = localPos1[:3]  # Convert to 3D vector

            # choose local Z axis in body0 and body1
            joint_prim.CreateAxisAttr().Set("Z")
            axis_Z = np.array([0, 0, 1])
            axis_in_Z0 = inv_trans0[:3, :3] @ axis
            axis_in_Z1 = inv_trans1[:3, :3] @ axis

            localRot0 = self._quat_from_veca_to_vecb(axis_Z, axis_in_Z0)
            localRot1 = self._quat_from_veca_to_vecb(axis_Z, axis_in_Z1)

            joint_prim.CreateLocalPos0Attr().Set(Gf.Vec3f(*localPos0))
            joint_prim.CreateLocalPos1Attr().Set(Gf.Vec3f(*localPos1))
            joint_prim.CreateLocalRot0Attr().Set(Gf.Quatf(*localRot0))
            joint_prim.CreateLocalRot1Attr().Set(Gf.Quatf(*localRot1))
            joint_prim.GetPrim()
            joint_prim.CreateBody0Rel().SetTargets([Sdf.Path(f"{self.root_prim.GetPath()}/{body_0_name}")])
            joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f"{self.root_prim.GetPath()}/{body_1_name}")])
            rbs_angle = joint_prim.GetPrim().CreateAttribute("rbs:angle", Sdf.ValueTypeNames.Float, custom=True)
            rbs_angle.Set(0.0)  # initial angle

            # limits
            urdf_joint = self.urdf_loader._joint_map.get(name, None)
            assert urdf_joint is not None, f"Joint {name} not found in URDF joint map."
            if urdf_joint.limit is not None:
                joint_prim.CreateLowerLimitAttr().Set(np.rad2deg(urdf_joint.limit.lower))
                joint_prim.CreateUpperLimitAttr().Set(np.rad2deg(urdf_joint.limit.upper))

        # 3) create fixed joints for root mesh link
        self._add_fixed_joint_to_root_mesh_link(self.urdf_loader.root_mesh_link_name)

    def _urdf_mesh_to_usd_mesh(self, usd_mesh: UsdGeom.Mesh, transform: np.ndarray, mesh_path: str) -> UsdGeom.Mesh:
        if mesh_path.startswith("package://"):
            mesh_path = mesh_path.replace("package://", "")
            mesh_path = f"{self.package_path}/{mesh_path}"
        elif pl.Path(mesh_path).is_absolute():
            mesh_path = pl.Path(mesh_path).resolve()
        else:
            mesh_path = pl.Path(self.urdf_folder, mesh_path).resolve()

        print(f"Loading mesh from: {mesh_path}")

        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_path}")

        # load the mesh using trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        usd_mesh.ClearXformOpOrder()
        usd_mesh.AddTransformOp().Set(Gf.Matrix4d(transform.T))
        usd_mesh.CreatePointsAttr(mesh.vertices)
        usd_mesh.CreateFaceVertexCountsAttr([len(face) for face in mesh.faces])
        usd_mesh.CreateFaceVertexIndicesAttr([int(index) for face in mesh.faces for index in face])
        return usd_mesh

    def _add_fixed_joint_to_root_mesh_link(self, mesh_link_name: str):
        # add a fixed joint to the root mesh link
        joint_prim = UsdPhysics.FixedJoint.Define(self.usd_stage, f"{self.root_prim.GetPath()}/joints/root_fixed_joint")
        joint_prim.CreateBody1Rel().SetTargets([Sdf.Path(f"{self.root_prim.GetPath()}/{mesh_link_name}")])
        return joint_prim

    @staticmethod
    def setup_stage(stage: Usd.Stage, up_axis: str = "Z", meters_per_unit: float = 1.0):
        # (
        #     endTimeCode = 0
        #     startTimeCode = -1
        #     upAxis = "Z"
        #     metersPerUnit = 1.0
        #     defaultPrim = "World"
        # )
        stage.SetStartTimeCode(-1)
        stage.SetEndTimeCode(0)
        UsdGeom.SetStageUpAxis(stage, up_axis)
        UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
        world = UsdGeom.Xform.Define(stage, "/World")
        world.AddTransformOp().Set(Gf.Matrix4d().SetIdentity())
        stage.SetDefaultPrim(world.GetPrim())
