"""比較動画の MP4 書き出し。

ブラウザが比較ビューを 1 コマずつ描画して JPEG で送り、ここで ffmpeg（H.264 / AAC）にそのまま流し込む。
ブラウザ内のエンコード（WebCodecs）は LAN 越しの http では使えず、Linux の Chrome は AAC も作れないため、
エンコードはサーバーで行う。コマは一時ファイルにせず、ffmpeg の標準入力に渡す。

    start(width, height, fps, frames, audio)  ffmpeg を起動する
    add_frame(id, jpeg)                        コマを順に渡す
    finish(id)                                 入力を閉じて MP4 を完成させる
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from .db import new_id

MAX_SECONDS = 60.0
SIZES = {(1920, 1080), (1080, 1920)}
FPS = 30
KEEP_SECONDS = 24 * 3600  # 書き出した MP4 は 1 日で消す


class ExportError(ValueError):
    pass


@dataclass
class ExportSession:
    id: str
    path: Path
    frames: int
    proc: subprocess.Popen
    received: int = 0
    log: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    started: float = field(default_factory=time.monotonic)


class ExportManager:
    def __init__(self, exports_dir: Path) -> None:
        self.dir = exports_dir
        self.sessions: dict[str, ExportSession] = {}
        self._lock = threading.Lock()

    def start(self, *, width: int, height: int, frames: int, audio: Path | None) -> ExportSession:
        if shutil.which("ffmpeg") is None:
            raise ExportError("ffmpeg がない")
        if (width, height) not in SIZES:
            raise ExportError(f"未対応の解像度: {width}x{height}")
        if not 1 <= frames <= int(MAX_SECONDS * FPS):
            raise ExportError(f"コマ数は 1〜{int(MAX_SECONDS * FPS)}（{MAX_SECONDS:.0f} 秒まで）")
        self.dir.mkdir(parents=True, exist_ok=True)
        self._cleanup()
        sid = new_id()
        path = self.dir / f"{sid}.mp4"
        duration = frames / FPS
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS), "-i", "-"]
        if audio is not None:
            # 音声が映像より短ければ無音で埋め、長ければ映像の長さで切る
            cmd += ["-i", str(audio), "-map", "0:v", "-map", "1:a", "-af", "apad", "-c:a", "aac", "-b:a", "192k"]
        cmd += [
            # JPEG はフルレンジなので、一般のプレーヤーや SNS 向けに通常のレンジの yuv420p にする
            "-vf", "scale=in_range=pc:out_range=tv,format=yuv420p",
            "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-color_range", "tv",
            "-t", f"{duration:.3f}", "-movflags", "+faststart", str(path),
        ]  # fmt: skip
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        session = ExportSession(id=sid, path=path, frames=frames, proc=proc)
        with self._lock:
            self.sessions[sid] = session
        return session

    def get(self, sid: str) -> ExportSession:
        with self._lock:
            session = self.sessions.get(sid)
        if session is None:
            raise KeyError(f"export not found: {sid}")
        return session

    def add_frame(self, sid: str, jpeg: bytes) -> int:
        session = self.get(sid)
        if not jpeg.startswith(b"\xff\xd8"):
            raise ExportError("JPEG ではない")
        with session.lock:
            if session.received >= session.frames:
                raise ExportError("コマが多すぎる")
            if session.proc.poll() is not None:
                raise ExportError(f"ffmpeg が終了している: {self._stderr(session)}")
            try:
                session.proc.stdin.write(jpeg)
            except BrokenPipeError:
                raise ExportError(f"ffmpeg が終了している: {self._stderr(session)}") from None
            session.received += 1
            return session.received

    def finish(self, sid: str) -> ExportSession:
        session = self.get(sid)
        with session.lock:
            if session.received != session.frames:
                raise ExportError(f"コマが足りない（{session.received} / {session.frames}）")
            session.proc.stdin.close()
            code = session.proc.wait(timeout=300)
            if code != 0 or not session.path.exists():
                raise ExportError(f"ffmpeg が失敗した: {self._stderr(session)}")
        return session

    def cancel(self, sid: str) -> None:
        session = self.get(sid)
        with session.lock:
            if session.proc.poll() is None:
                session.proc.kill()
                session.proc.wait()
            session.path.unlink(missing_ok=True)
        with self._lock:
            self.sessions.pop(sid, None)

    def _stderr(self, session: ExportSession) -> str:
        if session.proc.poll() is None:
            return ""
        return (session.proc.stderr.read() or b"").decode(errors="replace").strip()[-500:]

    def _cleanup(self) -> None:
        """古い MP4 と、終わったまま残っているセッションを消す。"""
        now = time.time()
        for f in self.dir.glob("*.mp4"):
            if now - f.stat().st_mtime > KEEP_SECONDS:
                f.unlink(missing_ok=True)
        with self._lock:
            for sid, s in list(self.sessions.items()):
                if not s.path.exists() and s.proc.poll() is not None:
                    self.sessions.pop(sid)
