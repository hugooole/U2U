# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025 zhiguo
import os.path as osp
import typing
from typing import Union, overload

import numpy as np
from loguru import logger
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics


def set_or_add_scale_op(
    xformable: UsdGeom.Xformable, scale: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h]
) -> UsdGeom.XformOp:
    """
    Sets or adds the scale XformOp on the input Xformable to provided scale value.

    Note that:
        - The precision of an added attribute is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        scale:      The scale vector
    Returns:
        The set or added XformOp
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_scale_op: Provided prim is not an Xformable")
        return False
    xformOp = _get_or_create_xform_op(xformable, "xformOp:scale", UsdGeom.XformOp.TypeScale)
    if xformOp.Get() is None:
        xformOp.Set(Gf.Vec3f(scale))
    else:
        typeName = type(xformOp.Get())
        xformOp.Set(typeName(scale))
    return xformOp


def set_or_add_translate_op(
    xformable: UsdGeom.Xformable, translate: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h]
) -> UsdGeom.XformOp:
    """
    Sets or adds the translate XformOp on the input Xformable to provided translate value.

    Note that:
        - The precision of an added attribute is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        translate:      The translate vector
    Returns:
        The set or added XformOp
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_translate_op: Provided prim is not an Xformable")
        return False
    xformOp = _get_or_create_xform_op(xformable, "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    if xformOp.Get() is None:
        xformOp.Set(Gf.Vec3f(translate))
    else:
        typeName = type(xformOp.Get())
        xformOp.Set(typeName(translate))
    return xformOp


def set_or_add_transform_op(
    xformable: UsdGeom.Xformable,
    transform: typing.Union[Gf.Matrix4d, Gf.Matrix4f],
    time: typing.Union[float, Usd.TimeCode, Sdf.TimeCode] = Usd.TimeCode.Default(),
):
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_transform_op: Provided prim {prim.GetPath()} is not an Xformable")
        return False
    xformOp = _get_or_create_xform_op(
        xformable,
        "xformOp:transform",
        UsdGeom.XformOp.TypeTransform,
        UsdGeom.XformOp.PrecisionDouble,
    )
    if xformOp.Get() is None:
        xformOp.Set(transform, time)
    else:
        # typeName = type(xformOp.Get())
        xformOp.Set(transform, time)
    return xformOp


def set_or_add_translate_op_with_time(
    xformable: UsdGeom.Xformable,
    translate: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h],
    time: typing.Union[float, Usd.TimeCode, Sdf.TimeCode] = Usd.TimeCode.Default(),
) -> UsdGeom.XformOp:
    """
    Sets or adds the translate XformOp on the input Xformable to provided translate value.

    Note that:
        - The precision of an added attribute is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        translate:      The translate vector
        time:       The time at which to set the translate value, defaults to Usd.TimeCode.Default()
    Returns:
        The set or added XformOp
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_translate_op: Provided prim is not an Xformable")
        return False
    xformOp = _get_or_create_xform_op(xformable, "xformOp:translate", UsdGeom.XformOp.TypeTranslate)
    if xformOp.Get() is None:
        xformOp.Set(Gf.Vec3f(translate), time)
    else:
        typeName = type(xformOp.Get())
        xformOp.Set(typeName(translate), time)
    return xformOp


def set_or_add_orient_op(
    xformable: UsdGeom.Xformable, orient: typing.Union[Gf.Quatf, Gf.Quatd, Gf.Quath]
) -> UsdGeom.XformOp:
    """
    Sets or adds the orient XformOp on the input Xformable to provided orient value.

    Note that:
        - The precision of an added attribute is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        orient:      The orient quaternion
    Returns:
        The set or added XformOp
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_orient_op: Provided prim is not an Xformable")
        return None
    xformOp = _get_or_create_xform_op(xformable, "xformOp:orient", UsdGeom.XformOp.TypeOrient)
    if xformOp.Get() is None:
        xformOp.Set(Gf.Quatf(orient))
    else:
        typeName = type(xformOp.Get())
        xformOp.Set(typeName(orient))
    return xformOp


def set_or_add_orient_op_with_time(
    xformable: UsdGeom.Xformable,
    orient: typing.Union[Gf.Quatf, Gf.Quatd, Gf.Quath],
    time: typing.Union[float, Usd.TimeCode, Sdf.TimeCode] = Usd.TimeCode.Default(),
) -> UsdGeom.XformOp:
    """
    Sets or adds the orient XformOp on the input Xformable to provided orient value.

    Note that:
        - The precision of an added attribute is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        orient:      The orient quaternion
        time:       Timecode at which to set the value
    Returns:
        The set or added XformOp
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_orient_op: Provided prim is not an Xformable")
        return None
    xformOp = _get_or_create_xform_op(xformable, "xformOp:orient", UsdGeom.XformOp.TypeOrient)
    if xformOp.Get() is None:
        xformOp.Set(Gf.Quatf(orient), time)
    else:
        typeName = type(xformOp.Get())
        xformOp.Set(typeName(orient), time)
    return xformOp


def set_or_add_orient_translate_with_time(
    xformable: UsdGeom.Xformable,
    orient: typing.Union[Gf.Quatf, Gf.Quatd, Gf.Quath],
    translate: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h],
    time: typing.Union[float, Usd.TimeCode, Sdf.TimeCode] = Usd.TimeCode.Default(),
) -> typing.List[UsdGeom.XformOp]:
    """
    Sets or adds orient and translate XformOps of xformable.

    Note that:
        - The precision of created attributes is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        orient:     The orientation quaternion
        translate:  The translation vector
        time:       Timecode at which to set the value
    Returns:
        List of set and created xform ops that will be [translate, orient]
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(
            f"{__name__}.set_or_add_orient_and_translate_with_time: Provided prim {prim.GetPath()} is not an Xformable"
        )
        return False
    tosOps = []
    tosOps.append(set_or_add_translate_op_with_time(xformable, translate, time))
    tosOps.append(set_or_add_orient_op_with_time(xformable, orient, time))
    return tosOps


def set_or_add_transform_with_time(
    xformable: UsdGeom.Xformable,
    transform: typing.Union[Gf.Matrix4d, Gf.Matrix4f],
    time: typing.Union[float, Usd.TimeCode, Sdf.TimeCode] = Usd.TimeCode.Default(),
) -> typing.List[UsdGeom.XformOp]:
    """
    Sets or adds scale, orient, and translate XformOps of xformable from a transform matrix.

    Note that:
        - The precision of created attributes is UsdGeom.XformOp.PrecisionFloat.
        - Any skew in the transform will be lost.

    Args:
        xformable:  The Xformable to modify.
        transform:  The transform matrix
        time:       Timecode at which to set the value
    Returns:
        List of set and created xform ops that will be [translate, orient, scale]
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_transform_with_time: Provided prim {prim.GetPath()} is not an Xformable")
        return False
    tosOps = []
    tosOps.append(set_or_add_transform_op(xformable, transform, time))
    return tosOps


def set_or_add_scale_orient_translate(
    xformable: UsdGeom.Xformable,
    scale: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h],
    orient: typing.Union[Gf.Quatf, Gf.Quatd, Gf.Quath],
    translate: typing.Union[Gf.Vec3f, Gf.Vec3d, Gf.Vec3h],
) -> typing.List[UsdGeom.XformOp]:
    """
    Sets or adds scale, orient, and translate XformOps of xformable.

    Note that:
        - The precision of created attributes is UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        scale:      The scale vector
        orient:     The orientation quaternion
        translate:  The translation vector
    Returns:
        List of set and created xform ops that will be [translate, orient, scale]
    """
    prim = xformable.GetPrim()
    if not (prim.IsA(UsdGeom.Xformable)):
        logger.warning(f"{__name__}.set_or_add_scale_orient_translate: Provided prim is not an Xformable")
        return False
    tosOps = []
    tosOps.append(set_or_add_translate_op(xformable, translate))
    tosOps.append(set_or_add_orient_op(xformable, orient))
    tosOps.append(set_or_add_scale_op(xformable, scale))
    return tosOps


def _get_or_create_xform_op(
    xformable: UsdGeom.Xformable,
    opName: str,
    opType: str,
    opPrecisionIfCreate=UsdGeom.XformOp.PrecisionFloat,
) -> UsdGeom.XformOp:
    """
    Gets or creates an XformOp of an Xformable.

    Note that:
        - Any skew in the transform will be lost.
        - A resetXformStack is preserved, but not the XformOps that are ignored due to the reset.
        - The transform attribute precision is set to UsdGeom.XformOp.PrecisionFloat.

    Args:
        xformable:  The Xformable to modify.
        opName:     The XformOp attribute name, e.g. "xformOp:translate"
        opType:     The XformOp type, e.g. UsdGeom.XformOp.TypeScale
    """
    dstOp = UsdGeom.XformOp(xformable.GetPrim().GetAttribute(opName))
    if not dstOp:
        # create op
        dstOp = xformable.AddXformOp(opType, opPrecisionIfCreate)
    return dstOp


def read_usd(path):
    if not osp.exists(path):
        raise FileNotFoundError(f"USD file does not exist: {path}")
    stage = Usd.Stage.Open(path)
    if not stage:
        raise RuntimeError(f"Failed to open USD file: {path}")
    return stage


def save_usd(stage: Usd.Stage, path: str):
    if not stage:
        raise ValueError("USD stage is not valid.")
    stage.GetRootLayer().Export(path)
    logger.info(f"Saved USD stage to {path}")


def get_or_create_rigid_body_api(prim: Usd.Prim):
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(prim)
    return UsdPhysics.RigidBodyAPI(prim)


def get_or_create_collision_api(prim: Usd.Prim):
    if not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
    return UsdPhysics.CollisionAPI(prim)


def get_prim_name(path) -> str:
    """Extract the short name (last segment) from a USD prim path.

    Args:
        path: A USD prim path (str or Sdf.Path).

    Returns:
        The last segment of the path, e.g. "/World/table" -> "table".
    """
    return str(path).split("/")[-1]


def get_prim_type_name(prim: Usd.Prim):
    return prim.GetPrimTypeInfo().GetTypeName()


def get_default_prim(stage: Usd.Stage, primPathAsString=True) -> Union[str, Sdf.Path]:
    """
    Get the default prim of the USD stage.
    """
    if not stage:
        raise ValueError("USD stage is not valid.")
    default_prim_path = stage.GetDefaultPrim().GetPath()
    if not default_prim_path:
        raise ValueError("No default prim found in the USD stage.")
    if primPathAsString:
        default_prim_path = str(default_prim_path)
    return default_prim_path


def has_attribute(prim: Usd.Prim, name: str) -> bool:
    """
    Check if a USD prim has a valid and authored attribute.

    Args:
        prim: The USD prim to query.
        name: The name of the attribute to check.

    Returns:
        True if the attribute exists, is valid, and has an authored value, False otherwise.
    """
    attr = prim.GetAttribute(name)
    return attr and attr.HasAuthoredValue()


@overload
def get_float(prim: Usd.Prim, name: str, default: float) -> float: ...


@overload
def get_float(prim: Usd.Prim, name: str, default: None = None) -> float | None: ...


def get_float(prim: Usd.Prim, name: str, default: float | None = None) -> float | None:
    """
    Get a float attribute value from a USD prim, validating that it's finite.

    Args:
        prim: The USD prim to query.
        name: The name of the float attribute to retrieve.
        default: The default value to return if the attribute is not found or is not finite.

    Returns:
        The float attribute value if it exists and is finite, otherwise the default value.
    """
    attr = prim.GetAttribute(name)
    if not attr or not attr.HasAuthoredValue():
        return default
    val = attr.Get()
    if np.isfinite(val):
        return val
    return default
