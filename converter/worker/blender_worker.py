"""共通コンバーター: Blender に常駐し、HTTP で変換ジョブを受け付ける。

起動:
    blender -b --python converter/worker/blender_worker.py -- --stickman assets/stickman/stickman.vrm --port 8100

API（Blender の bpy はスレッドセーフでないため、メインスレッドで 1 件ずつ処理する）:
    GET  /health   -> {"status": "ready"}
    POST /convert  {"source_motion": path, "bone_map": {...}, "out_dir": path,
                    "preview": true, "bvh": true}
                -> {"files": {"vrma": "motion.vrma", "preview": "preview.webp", "bvh": "debug.bvh"}, ...}
    POST /preview_group {"items": [{"source_motion", "bone_map", "color"}], "out": path}
                -> {"file": "group_preview.webp"}

処理:
    1. SourceMotion を読み、sourcemotion.retarget で棒人間の VRM 正規化ボーンへリターゲット（30fps）
    2. 棒人間アーマチュアにキーフレームとして書き込み、VRM Add-on で VRMA を書き出す
    3. 確認用 BVH とプレビュー WebP（Workbench で描画し ffmpeg でまとめる）を書き出す
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import bpy
import numpy as np
from bpy_extras import anim_utils
from mathutils import Matrix, Quaternion, Vector

from sourcemotion import SourceMotion, quat
from sourcemotion.retarget import RetargetResult, retarget
from sourcemotion.vrm import load_vrm_humanoid

ADDON_MODULE = "bl_ext.user_default.vrm"
PREVIEW_FPS = 15
PREVIEW_SIZE = 320

# glTF（+Y 上、+Z 前方）→ Blender（+Z 上、-Y 前方）: X 軸まわりに +90 度
GLTF_TO_BLENDER = quat.from_axis_angle(np.array([1.0, 0.0, 0.0]), np.array(math.pi / 2))


def snake(name: str) -> str:
    return "".join("_" + c.lower() if c.isupper() else c for c in name)


def to_xyzw(q: Quaternion) -> np.ndarray:
    return np.array([q.x, q.y, q.z, q.w], dtype=np.float64)


class Rig:
    """読み込んだ棒人間 1 体（アーマチュアとメッシュ）。"""

    def __init__(self, arm: bpy.types.Object, vrm_bones: list[str]) -> None:
        self.arm = arm
        self.meshes = [o for o in arm.children if o.type == "MESH"]
        human_bones = arm.data.vrm_addon_extension.vrm1.humanoid.human_bones
        self.bone_names: dict[str, str] = {}
        for vrm_bone in vrm_bones:
            prop = getattr(human_bones, snake(vrm_bone), None)
            name = prop.node.bone_name if prop is not None else ""
            if name:
                self.bone_names[vrm_bone] = name
        if "hips" not in self.bone_names:
            raise RuntimeError("hips bone not found in stickman")
        # VRM の読み込み後は hips が root に接続（use_connect）されていて移動できないため、接続を外す
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.mode_set(mode="EDIT")
        arm.data.edit_bones[self.bone_names["hips"]].use_connect = False
        bpy.ops.object.mode_set(mode="OBJECT")
        for pb in arm.pose.bones:
            pb.rotation_mode = "QUATERNION"

    def set_visible(self, visible: bool) -> None:
        for o in (self.arm, *self.meshes):
            o.hide_render = not visible
            o.hide_viewport = not visible

    def clear(self) -> None:
        for owner in (self.arm, self.arm.data):  # 表情のアクションはアーマチュアのデータ側に付く
            if owner.animation_data and owner.animation_data.action:
                old = owner.animation_data.action
                owner.animation_data.action = None
                bpy.data.actions.remove(old)

    def apply(self, result: RetargetResult) -> None:
        arm = self.arm
        mw = arm.matrix_world
        # glTF ワールド → アーマチュア空間
        m_q = quat.multiply(to_xyzw(mw.to_quaternion().inverted()), GLTF_TO_BLENDER)
        m_inv = quat.inverse(m_q)

        self.clear()
        write = _new_action(arm, "OBJECT", "vmb_motion", result.num_frames)
        for k, vrm_bone in enumerate(result.bones):
            name = self.bone_names.get(vrm_bone)
            if name is None:
                continue
            bone = arm.data.bones[name]
            rb = to_xyzw(bone.matrix_local.to_quaternion())
            local = quat.multiply(quat.multiply(m_q, result.local_rotations[:, k].astype(np.float64)), m_inv)
            basis = quat.multiply(quat.multiply(quat.inverse(rb)[None], local), rb[None])
            basis = quat.make_continuous(basis)
            write(f'pose.bones["{name}"].rotation_quaternion', basis[:, [3, 0, 1, 2]])  # wxyz
            if vrm_bone == "hips":
                head = quat.rotate(m_q, result.hips_translation.astype(np.float64))
                head -= np.array(mw.translation)
                loc = quat.rotate(quat.inverse(rb)[None], head - np.array(bone.head_local))
                write(f'pose.bones["{name}"].location', loc)
        if result.expressions:
            self._apply_expressions(result)

    def _apply_expressions(self, result: RetargetResult) -> None:
        """表情の重みを VRM Add-on の表情プレビュー値のキーにする。VRMA の書き出し時に表情トラックになる。"""
        expressions = self.arm.data.vrm_addon_extension.vrm1.expressions.all_name_to_expression_dict()
        known = {name: expr for name, expr in expressions.items() if name in result.expressions}
        if not known:
            return
        write = _new_action(self.arm.data, "ARMATURE", "vmb_expressions", result.num_frames)
        for name, expr in known.items():
            write(expr.path_from_id("preview"), result.expressions[name].astype(np.float64)[:, None])


def _new_action(owner: bpy.types.ID, id_type: str, name: str, n: int):
    """owner に新しいアクションを付け、「データパスと [n, k] の値から LINEAR のキーを打つ関数」を返す。"""
    if owner.animation_data is None:
        owner.animation_data_create()
    action = bpy.data.actions.new(name)
    slot = action.slots.new(id_type=id_type, name=owner.name)
    owner.animation_data.action = action
    owner.animation_data.action_slot = slot
    bag = anim_utils.action_ensure_channelbag_for_slot(action, slot)
    frames = np.arange(1, n + 1, dtype=np.float64)

    def write(data_path: str, values: np.ndarray) -> None:
        for i in range(values.shape[1]):
            fc = bag.fcurves.new(data_path, index=i)
            fc.keyframe_points.add(n)
            co = np.empty((n, 2), dtype=np.float64)
            co[:, 0], co[:, 1] = frames, values[:, i]
            fc.keyframe_points.foreach_set("co", co.ravel())
            fc.keyframe_points.foreach_set("interpolation", [1] * n)  # LINEAR
            fc.update()

    return write


class Converter:
    def __init__(self, stickman: Path) -> None:
        self.stickman = stickman
        self.target = load_vrm_humanoid(stickman)
        bpy.ops.preferences.addon_enable(module=ADDON_MODULE)
        bpy.ops.wm.read_homefile(use_empty=True)
        self.rigs = [self._import_rig()]
        self._setup_render()

    @property
    def arm(self) -> bpy.types.Object:
        return self.rigs[0].arm

    def _import_rig(self) -> Rig:
        before = set(bpy.data.objects)
        # 描画エンジンが Workbench のままだと、VRM Add-on が読み込み時にビューポートを
        # 「マテリアル表示」へ切り替えようとして失敗するため、読み込みの間だけ戻す
        scene = bpy.context.scene
        engine = scene.render.engine
        scene.render.engine = "BLENDER_EEVEE"
        try:
            result = bpy.ops.import_scene.vrm(filepath=str(self.stickman))
        finally:
            scene.render.engine = engine
        if result != {"FINISHED"}:
            raise RuntimeError(f"stickman import failed: {result}")
        arm = next(o for o in bpy.data.objects if o not in before and o.type == "ARMATURE")
        return Rig(arm, self.target.bones)

    # ------------------------------------------------------------ 変換

    def convert(self, req: dict) -> dict:
        started = time.monotonic()
        out_dir = Path(req["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        motion = SourceMotion.load(req["source_motion"])
        result = retarget(motion, req["bone_map"], self.target)
        self._apply(result)

        files: dict[str, str] = {}
        vrma = out_dir / "motion.vrma"
        r = bpy.ops.export_scene.vrma(filepath=str(vrma), armature_object_name=self.arm.name)
        if r != {"FINISHED"} or not vrma.exists():
            raise RuntimeError(f"VRMA export failed: {r}")
        files["vrma"] = vrma.name
        if req.get("bvh", True):
            files["bvh"] = self._export_bvh(out_dir / "debug.bvh", result.num_frames).name
        if result.contacts:
            # 接地フラグ（30fps、VRM ボーンごとに "0"/"1" の列）。ビューアの接地マーカーに使う
            contacts = {
                "fps": result.fps,
                "bones": {b: "".join("1" if f else "0" for f in flags) for b, flags in result.contacts.items()},
            }
            (out_dir / "contacts.json").write_text(json.dumps(contacts))
            files["contacts"] = "contacts.json"
        if req.get("preview", True):
            files["preview"] = self._render_preview(out_dir / "preview.webp", [result]).name
        return {
            "files": files,
            "frames": result.num_frames,
            "fps": result.fps,
            "duration": (result.num_frames - 1) / result.fps,
            "hips_scale": result.hips_scale,
            "mapped_bones": sorted(result.mapped),
            "expressions": sorted(result.expressions or {}),
            "seconds": time.monotonic() - started,
        }

    def preview_group(self, req: dict) -> dict:
        """複数の結果を色分け・半透明で重ねたプレビュー。短いクリップは最後のフレームで止まる。

        req: {"items": [{"source_motion", "bone_map", "color": "#rrggbb"}], "out": path}
        """
        started = time.monotonic()
        items = req["items"]
        while len(self.rigs) < len(items):
            self.rigs.append(self._import_rig())
        results = []
        for rig, item in zip(self.rigs, items):
            result = retarget(SourceMotion.load(item["source_motion"]), item["bone_map"], self.target)
            rig.apply(result)
            rig.set_visible(True)
            color = item.get("color", "#cccccc").lstrip("#")
            rgb = [int(color[i : i + 2], 16) / 255 for i in (0, 2, 4)]
            for mesh in rig.meshes:
                mesh.color = (*rgb, 1.0)
            results.append(result)
        for rig in self.rigs[len(items) :]:
            rig.set_visible(False)

        shading = bpy.context.scene.display.shading
        shading.color_type, shading.show_xray, shading.xray_alpha = "OBJECT", True, 0.65
        self._set_frames(max(r.num_frames for r in results), results[0].fps)
        try:
            out = self._render_preview(Path(req["out"]), results)
        finally:
            shading.color_type, shading.show_xray = "MATERIAL", False
            for rig in self.rigs[1:]:
                rig.clear()
                rig.set_visible(False)
            self.rigs[0].set_visible(True)
        return {"file": out.name, "seconds": time.monotonic() - started}

    def _apply(self, result: RetargetResult) -> None:
        for rig in self.rigs[1:]:
            rig.set_visible(False)
        self.rigs[0].apply(result)
        self._set_frames(result.num_frames, result.fps)

    def _set_frames(self, n: int, fps: float) -> None:
        scene = bpy.context.scene
        scene.render.fps = int(round(fps))
        scene.render.fps_base = 1.0
        scene.frame_start, scene.frame_end = 1, n
        scene.frame_set(1)

    def _export_bvh(self, path: Path, n: int) -> Path:
        bpy.ops.object.select_all(action="DESELECT")
        self.arm.select_set(True)
        bpy.context.view_layer.objects.active = self.arm
        bpy.ops.export_anim.bvh(filepath=str(path), frame_start=1, frame_end=n, global_scale=1.0)
        return path

    # ------------------------------------------------------------ プレビュー

    def _setup_render(self) -> None:
        scene = bpy.context.scene
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.display.shading.light = "STUDIO"
        scene.display.shading.color_type = "MATERIAL"
        scene.render.resolution_x = scene.render.resolution_y = PREVIEW_SIZE
        scene.render.resolution_percentage = 100
        scene.render.film_transparent = False
        scene.render.image_settings.file_format = "PNG"
        if scene.world is None:
            scene.world = bpy.data.worlds.new("world")
        scene.world.color = (0.16, 0.17, 0.19)

        # 床（0.5m の市松模様）。重ね合わせ時はオブジェクト色で塗るため、色ごとに別オブジェクトにする
        tiles, size = 40, 0.5
        self.floor = bpy.data.objects.new("floor", None)
        bpy.context.scene.collection.objects.link(self.floor)
        for k, rgba in enumerate([(0.30, 0.31, 0.33, 1), (0.36, 0.37, 0.40, 1)]):
            verts, faces = [], []
            for i in range(tiles):
                for j in range(tiles):
                    if (i + j) % 2 != k:
                        continue
                    x0, y0 = (i - tiles / 2) * size, (j - tiles / 2) * size
                    base = len(verts)
                    verts += [(x0, y0, 0), (x0 + size, y0, 0), (x0 + size, y0 + size, 0), (x0, y0 + size, 0)]
                    faces.append((base, base + 1, base + 2, base + 3))
            mesh = bpy.data.meshes.new(f"floor_{k}")
            mesh.from_pydata(verts, [], faces)
            mat = bpy.data.materials.new(f"floor_{k}")
            mat.diffuse_color = rgba
            mesh.materials.append(mat)
            tile = bpy.data.objects.new(f"floor_{k}", mesh)
            tile.color = rgba
            tile.parent = self.floor
            bpy.context.scene.collection.objects.link(tile)

        cam_data = bpy.data.cameras.new("preview_cam")
        cam_data.type = "ORTHO"
        self.camera = bpy.data.objects.new("preview_cam", cam_data)
        bpy.context.scene.collection.objects.link(self.camera)
        scene.camera = self.camera

    def _render_preview(self, path: Path, results: list[RetargetResult]) -> Path:
        scene = bpy.context.scene
        n = max(r.num_frames for r in results)
        fps = results[0].fps
        # カメラは腰の水平位置（複数なら平均）を追う。短いクリップは最後の位置で止まる
        hips = np.zeros((n, 3))
        for r in results:
            h = quat.rotate(GLTF_TO_BLENDER, r.hips_translation.astype(np.float64))
            hips += np.concatenate([h, np.repeat(h[-1:], n - len(h), axis=0)])
        hips /= len(results)
        follow = hips[:, :2].copy()
        for k in range(1, len(follow)):  # 上下動や細かい揺れを追わないよう平滑化する
            follow[k] = follow[k - 1] * 0.85 + follow[k] * 0.15
        follow -= follow[0] - hips[0, :2]
        lo, hi = hips.min(axis=0), hips.max(axis=0)
        self.floor.location = (round((lo[0] + hi[0]) / 2), round((lo[1] + hi[1]) / 2), 0)
        cam = self.camera
        cam.data.ortho_scale = 2.6 if len(results) == 1 else 3.2
        cam.data.clip_end = 100.0
        direction = Vector((math.sin(math.radians(30)), -math.cos(math.radians(30)), 0.45)).normalized()

        step = max(1, int(round(fps / PREVIEW_FPS)))
        with tempfile.TemporaryDirectory() as tmp:
            for k, frame in enumerate(range(1, n + 1, step)):
                center = Vector((follow[frame - 1, 0], follow[frame - 1, 1], 0.85))
                cam.location = center + direction * 10.0
                cam.rotation_euler = (center - cam.location).to_track_quat("-Z", "Y").to_euler()
                scene.frame_set(frame)
                scene.render.filepath = str(Path(tmp) / f"{k:05d}.png")
                bpy.ops.render.render(write_still=True)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{fps / step:g}",
                "-i", str(Path(tmp) / "%05d.png"),
                "-c:v", "libwebp_anim", "-loop", "0", "-quality", "70", "-an", str(path),
            ]  # fmt: skip
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(cmd, check=True)
        return path


def make_handler(converter: Converter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._send(200, {"status": "ready", "blender": bpy.app.version_string})
            else:
                self._send(404, {"error_code": "not_found", "message": self.path})

        def do_POST(self) -> None:  # noqa: N802
            routes = {"/convert": converter.convert, "/preview_group": converter.preview_group}
            if self.path not in routes:
                self._send(404, {"error_code": "not_found", "message": self.path})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(length))
                self._send(200, routes[self.path](req))
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._send(500, {"error_code": "internal", "message": f"{type(e).__name__}: {e}"})

        def log_message(self, fmt: str, *args) -> None:
            print("[converter]", fmt % args, flush=True)

    return Handler


def main() -> None:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--stickman", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--once", help="サーバーを立てずに 1 件だけ変換する（JSON リクエストのパス）")
    args = parser.parse_args(argv)
    if shutil.which("ffmpeg") is None:
        print("[converter] warning: ffmpeg not found; previews will fail", flush=True)

    converter = Converter(Path(args.stickman))
    if args.once:
        print(json.dumps(converter.convert(json.loads(Path(args.once).read_text())), indent=2))
        return
    server = HTTPServer((args.host, args.port), make_handler(converter))
    print(f"[converter] ready on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
