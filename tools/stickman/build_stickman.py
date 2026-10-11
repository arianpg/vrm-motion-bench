"""VRM Motion Bench 用の棒人間 VRM 1.0 を生成する。

使い方:
    blender -b --python-use-system-env --python tools/stickman/build_stickman.py -- \
        --out assets/stickman/stickman.vrm

前提: Blender 5.0 + VRM Add-on for Blender 4.7 系（bl_ext.user_default.vrm）。
add-on は numpy を必要とするため、Blender の Python から numpy が見えること。

構成:
    - VRM Add-on の基本アーマチュア（T ポーズ、VRM 1.0 ヒューマノイド割り当て済み）
    - 各ボーンに剛体ウェイト（1.0）の円柱＋関節球を割り当てた 1 メッシュ
    - 色分け: キャラクターの左 = 青、右 = 赤、中央 = 灰、顔の向き = 黄
"""

import argparse
import math
import sys

import bpy
import bmesh
from mathutils import Matrix, Vector

ADDON_MODULE = "bl_ext.user_default.vrm"

TALL = 1.70
LEG_WIDTH_RATIO = 1.7  # 既定(1.0)だと股関節の間隔が約 0.1m と狭いため

COLORS = {
    "center": (0.62, 0.64, 0.68, 1.0),
    "left": (0.15, 0.42, 0.95, 1.0),
    "right": (0.92, 0.22, 0.20, 1.0),
    "front": (0.98, 0.80, 0.10, 1.0),
}

# ボーン名 -> 円柱半径。ここに無いボーン（root、eye）はメッシュを作らない
LIMB_RADIUS = {
    "hips": 0.050,
    "spine": 0.045,
    "chest": 0.050,
    "neck": 0.025,
    "shoulder": 0.022,
    "upper_arm": 0.030,
    "lower_arm": 0.025,
    "hand": 0.022,
    "upper_leg": 0.040,
    "lower_leg": 0.032,
    "foot": 0.028,
    "toes": 0.022,
}
FINGER_RADIUS = 0.008
HEAD_RADIUS = 0.095


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--blend", help="確認用に .blend も保存する場合のパス")
    return parser.parse_args(argv)


def side_of(bone_name: str) -> str:
    if bone_name.endswith(".L"):
        return "left"
    if bone_name.endswith(".R"):
        return "right"
    return "center"


def base_name(bone_name: str) -> str:
    return bone_name.rsplit(".", 1)[0] if bone_name[-2:] in (".L", ".R") else bone_name


def make_material(name: str, rgba: tuple[float, float, float, float]) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    # ノード名は UI 言語で翻訳されるため型で探す
    bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
    bsdf.inputs["Base Color"].default_value = rgba
    bsdf.inputs["Roughness"].default_value = 0.7
    bsdf.inputs["Metallic"].default_value = 0.0
    mat.diffuse_color = rgba
    return mat


class PartBuilder:
    """bmesh にパーツを追加し、ボーンのウェイト（変形レイヤー）とマテリアル番号を付ける。

    bmesh の演算は頂点の統合などで番号が変わることがあるため、頂点番号は使わず、
    「まだパーツ番号が付いていない頂点」を新しく作られた頂点とみなす。
    """

    def __init__(self, material_index: dict[str, int]) -> None:
        self.bm = bmesh.new()
        self.deform = self.bm.verts.layers.deform.verify()
        self.part = self.bm.verts.layers.int.new("part")
        self.next_part = 1
        self.groups: dict[str, int] = {}  # ボーン名 -> 頂点グループ番号
        self.material_index = material_index

    def _claim_new(self, bone: str, color: str) -> None:
        group = self.groups.setdefault(bone, len(self.groups))
        verts = [v for v in self.bm.verts if v[self.part] == 0]
        for v in verts:
            v[self.part] = self.next_part
            v[self.deform][group] = 1.0
        for f in {f for v in verts for f in v.link_faces}:
            f.material_index = self.material_index[color]
            f.smooth = True
        self.next_part += 1

    def cylinder(self, head: Vector, tail: Vector, radius: float, bone: str, color: str) -> None:
        axis = tail - head
        length = axis.length
        if length < 1e-6:
            return
        rot = axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
        mat = Matrix.Translation((head + tail) / 2) @ rot
        bmesh.ops.create_cone(
            self.bm, cap_ends=True, segments=12, radius1=radius, radius2=radius,
            depth=length, matrix=mat,
        )
        self._claim_new(bone, color)

    def sphere(self, center: Vector, radius: float, bone: str, color: str) -> None:
        bmesh.ops.create_uvsphere(
            self.bm, u_segments=16, v_segments=10, radius=radius,
            matrix=Matrix.Translation(center),
        )
        self._claim_new(bone, color)

    def cone(self, base: Vector, tip: Vector, radius: float, bone: str, color: str) -> None:
        axis = tip - base
        rot = axis.to_track_quat("Z", "Y").to_matrix().to_4x4()
        mat = Matrix.Translation((base + tip) / 2) @ rot
        bmesh.ops.create_cone(
            self.bm, cap_ends=True, segments=12, radius1=radius, radius2=0.0,
            depth=axis.length, matrix=mat,
        )
        self._claim_new(bone, color)


def build_mesh(arm_obj: bpy.types.Object) -> bpy.types.Object:
    bones = arm_obj.data.bones
    materials = {key: make_material(f"stickman_{key}", rgba) for key, rgba in COLORS.items()}
    material_index = {key: i for i, key in enumerate(materials)}
    pb = PartBuilder(material_index)

    for bone in bones:
        name = bone.name
        base = base_name(name)
        color = side_of(name)
        head, tail = bone.head_local.copy(), bone.tail_local.copy()

        if base == "head":
            center = head + (tail - head) * 0.55
            pb.sphere(center, HEAD_RADIUS, name, "center")
            # 顔の向き（VRM 1.0 は +Z 前方 = Blender では -Y）を示す鼻
            nose_base = center + Vector((0.0, -HEAD_RADIUS * 0.85, 0.0))
            pb.cone(nose_base, nose_base + Vector((0.0, -0.06, 0.0)), 0.025, name, "front")
            continue
        if base == "hips":
            continue  # 骨盤は下で個別に作る（hips の起点は股関節より下にあるため）

        is_finger = any(base.startswith(f) for f in ("thumb", "index", "middle", "ring", "little"))
        if is_finger:
            radius = FINGER_RADIUS
        elif base in LIMB_RADIUS:
            radius = LIMB_RADIUS[base]
        else:
            continue  # root, eye

        pb.cylinder(head, tail, radius, name, color)
        pb.sphere(head, radius * (1.0 if color == "center" else 1.15), name, color)
        if not bone.children:
            pb.sphere(tail, radius * 1.05, name, color)

    # 骨盤: 左右の股関節をつなぐ横棒と、その中点から spine 起点への縦棒
    hip_l = bones["upper_leg.L"].head_local
    hip_r = bones["upper_leg.R"].head_local
    pelvis_mid = (hip_l + hip_r) / 2
    pb.cylinder(hip_r, hip_l, LIMB_RADIUS["hips"] * 0.8, "hips", "center")
    pb.cylinder(pelvis_mid, bones["hips"].tail_local, LIMB_RADIUS["hips"], "hips", "center")
    pb.sphere(pelvis_mid, LIMB_RADIUS["hips"], "hips", "center")
    # 胸の前面マーカー（体幹の向き）
    chest = bones["chest"]
    chest_mid = (chest.head_local + chest.tail_local) / 2
    pb.cone(
        chest_mid + Vector((0.0, -0.045, 0.0)), chest_mid + Vector((0.0, -0.09, 0.0)),
        0.02, "chest", "front",
    )

    mesh = bpy.data.meshes.new("stickman_body")
    pb.bm.verts.layers.int.remove(pb.part)
    pb.bm.to_mesh(mesh)
    pb.bm.free()
    for mat in materials.values():
        mesh.materials.append(mat)

    obj = bpy.data.objects.new("stickman_body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    # 変形レイヤーのグループ番号の順に頂点グループを作る
    for bone_name, _ in sorted(pb.groups.items(), key=lambda kv: kv[1]):
        obj.vertex_groups.new(name=bone_name)

    obj.parent = arm_obj
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_obj
    return obj


def set_meta(arm_obj: bpy.types.Object) -> None:
    meta = arm_obj.data.vrm_addon_extension.vrm1.meta
    meta.vrm_name = "VRM Motion Bench Stickman"
    meta.version = "1.0.0"
    while len(meta.authors) > 0:
        meta.authors.remove(0)
    meta.authors.add().value = "arianpg"
    meta.avatar_permission = "everyone"
    meta.commercial_usage = "corporation"
    meta.credit_notation = "unnecessary"
    meta.allow_redistribution = True
    meta.modification = "allowModificationRedistribution"


def main() -> None:
    args = parse_args()
    bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
    bpy.ops.wm.read_homefile(use_empty=True)

    result = bpy.ops.icyp.make_basic_armature(tall=TALL, nail_bone=False, leg_width_ratio=LEG_WIDTH_RATIO)
    if result != {"FINISHED"}:
        raise RuntimeError(f"make_basic_armature failed: {result}")
    arm_obj = next(o for o in bpy.data.objects if o.type == "ARMATURE")
    arm_obj.name = arm_obj.data.name = "stickman_armature"
    if arm_obj.data.vrm_addon_extension.spec_version != "1.0":
        raise RuntimeError("armature is not VRM 1.0")

    build_mesh(arm_obj)
    set_meta(arm_obj)

    if args.blend:
        bpy.ops.wm.save_as_mainfile(filepath=args.blend)

    bpy.ops.object.select_all(action="DESELECT")
    arm_obj.select_set(True)
    bpy.context.view_layer.objects.active = arm_obj
    result = bpy.ops.export_scene.vrm(filepath=args.out, ignore_warning=True)
    if result != {"FINISHED"}:
        raise RuntimeError(f"VRM export failed: {result}")
    print(f"exported: {args.out}")


if __name__ == "__main__":
    main()
