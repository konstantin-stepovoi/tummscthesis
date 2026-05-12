from __future__ import annotations

from pathlib import Path
import threading
from typing import Any
from datetime import datetime
import traceback

from droplet_platform.acquisition import CameraConfig, BaslerCameraSource, VideoFileSource
from droplet_platform.measurement import record_droplet_intensities
from .preview import PreviewConfig, PreviewController
from .widgets import parse_ids


def _set_if_present(obj: Any, name: str, value: Any) -> bool:
    if hasattr(obj, name):
        try:
            current = getattr(obj, name)
            if isinstance(current, int) and not isinstance(current, bool):
                value = int(value)
            elif isinstance(current, float):
                value = float(value)
            setattr(obj, name, value)
            return True
        except Exception:
            return False
    return False


def update_detector_from_controls(detector, controls: dict) -> list[str]:
    target = getattr(detector, "base", detector)
    changed = []
    mapping = {
        "percentile": float(controls["percentile"].value),
        "ds": int(controls["downscale_factor"].value),
        "min_distance": int(controls["min_distance"].value),
        "corr_backend": str(controls["corr_backend"].value),
        "cpu_downscale": bool(controls["cpu_downscale"].value),
        "peak_downscale": int(controls["peak_downscale"].value),
    }
    aliases = {"ds": ["ds", "downscale_factor"]}
    for name, value in mapping.items():
        for candidate in aliases.get(name, [name]):
            if _set_if_present(target, candidate, value):
                changed.append(candidate)
                break
    radius = int(controls["droplet_radius_px"].value)
    if hasattr(target, "r") and hasattr(target, "tmpl_size"):
        try:
            ds = int(getattr(target, "ds", 1))
            target.r = int(radius // max(1, ds))
            target.tmpl_size = int(2 * target.r + 1)
            if hasattr(target, "_prepare_template"):
                tmpl = target._prepare_template(template_path=None, template_array=None, template_resize="auto")
                try:
                    import cupy as cp
                    target.tmpl_gpu = cp.asarray(tmpl, dtype=cp.float32)
                    target.tmpl_flip_gpu = target.tmpl_gpu[::-1, ::-1]
                except Exception:
                    pass
            changed.append("droplet_radius_px/template")
        except Exception:
            changed.append("radius-change-pending-restart")
    return changed


def bind_basic_callbacks(controls: dict, state, detector):
    """Bind widgets and route all background exceptions into the notebook log box."""
    holder = {"preview": None, "source": None, "recorder": None}

    def append_log(msg: str, *, also_file: bool = True):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}".rstrip()
        if "log" in controls:
            old = str(controls["log"].value or "")
            controls["log"].value = (old + "\n" + line).strip()[-12000:]
        print(line)
        if also_file:
            try:
                root = Path(str(controls["root"].value).strip() or "experiments").expanduser()
                root.mkdir(parents=True, exist_ok=True)
                with (root / "ui_debug.log").open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                pass

    def log_exception(prefix: str, exc: BaseException | None = None, tb: str | None = None):
        if tb is None and exc is not None:
            tb = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        text = f"{prefix}\n{tb or ''}".rstrip()
        state.set_error(tb or str(exc) if exc is not None else text)
        append_log(text)

    def set_status(msg: str):
        state.last_status = msg
        controls["status"].value = f"<b>Status:</b> {msg}"
        append_log(msg, also_file=False)

    def root_dir() -> Path:
        root = Path(str(controls["root"].value).strip() or "experiments").expanduser()
        root.mkdir(parents=True, exist_ok=True)
        state.root_dir = root
        return root

    def make_source():
        kind = str(controls["source_kind"].value)
        state.last_source_kind = kind
        if kind == "video":
            path = Path(str(controls["video_path"].value).strip().strip('"')).expanduser()
            if not str(path):
                raise RuntimeError("Video source selected, but video path is empty")
            if not path.exists():
                raise FileNotFoundError(f"Video file does not exist: {path}")
            fps = float(controls["fps"].value) if bool(controls["fake_realtime"].value) else None
            append_log(f"using video source: {path}")
            return VideoFileSource(path, loop=bool(controls["video_loop"].value), realtime_fps=fps)
        cfg = CameraConfig(fps=float(controls["fps"].value), exposure_ms=float(controls["exposure_ms"].value))
        append_log("using Basler camera source")
        return BaslerCameraSource(cfg)

    def update_finish_zone():
        fav = int(controls["favorite_id"].value)
        state.favorite_zone.favorite_id = fav if fav > 0 else None
        state.favorite_zone.enabled = bool(controls["finish_enabled"].value) and state.favorite_zone.favorite_id is not None
        state.favorite_zone.p0 = (int(controls["finish_x0"].value), int(controls["finish_y0"].value))
        state.favorite_zone.p1 = (int(controls["finish_x1"].value), int(controls["finish_y1"].value))
        state.favorite_zone.width_px = int(controls["finish_width"].value)

    def preview_config() -> PreviewConfig:
        root = root_dir()
        raw = root / str(controls["raw_video_name"].value) if bool(controls["record_raw_video"].value) else None
        overlay = root / str(controls["overlay_video_name"].value) if bool(controls["record_overlay_video"].value) else None
        return PreviewConfig(
            colorize=bool(controls["colorize"].value),
            draw_ids=bool(controls["draw_ids"].value),
            draw_detections=bool(controls["draw_detections"].value),
            beep_on_crossing=bool(controls["beep"].value),
            reset_tracker_on_start=bool(controls["reset_tracker"].value),
            record_raw_video_path=raw,
            record_overlay_video_path=overlay,
            record_fps=float(controls["fps"].value),
        )

    def on_apply_detector(_=None):
        try:
            changed = update_detector_from_controls(detector, controls)
            set_status("updated detector: " + ", ".join(changed or ["no compatible fields found"]))
        except Exception as exc:
            log_exception("detector update failed", exc)
            set_status("detector update failed; see log")

    def on_start(_):
        try:
            on_apply_detector()
            update_finish_zone()
            source = make_source()
            holder["source"] = source
            pc = PreviewController(source, detector, state, preview_config())
            pc.on_error = lambda exc, tb: log_exception("preview crashed", exc, tb)
            pc.on_status = set_status
            holder["preview"] = pc
            pc.start_background()
            set_status("preview/stream running")
        except Exception as exc:
            log_exception("preview start failed", exc)
            set_status("preview start failed; see log")

    def on_stop(_):
        try:
            if holder.get("preview") is not None:
                holder["preview"].stop()
            if holder.get("recorder") is not None:
                holder["recorder"].stop()
            state.stop_event.set()
            set_status("stop requested")
        except Exception as exc:
            log_exception("stop failed", exc)
            set_status("stop failed; see log")

    def on_snapshot_raw(_):
        try:
            out = root_dir() / f"snapshot_raw_{state.frame_id:06d}.png"
            state.save_last_frame(out, display=False)
            set_status(f"saved {out}")
        except Exception as exc:
            log_exception("raw snapshot failed", exc)
            set_status("snapshot failed; see log")

    def on_snapshot_overlay(_):
        try:
            out = root_dir() / f"snapshot_overlay_{state.frame_id:06d}.png"
            state.save_last_frame(out, display=True)
            set_status(f"saved {out}")
        except Exception as exc:
            log_exception("overlay snapshot failed", exc)
            set_status("snapshot failed; see log")

    def on_record(_):
        def worker():
            try:
                on_apply_detector()
                update_finish_zone()
                state.is_recording = True
                set_status("recording droplet intensities")
                source = make_source()
                root = root_dir()
                out = root / str(controls["output_name"].value)
                ids = parse_ids(str(controls["ids"].value)) or state.active_ids()
                path = record_droplet_intensities(
                    source=source,
                    detector=detector,
                    tracker=state.tracker,
                    output_path=out,
                    duration_s=float(controls["duration_s"].value),
                    fps=float(controls["fps"].value),
                    radius_px=int(controls["roi_radius_px"].value),
                    ids_to_track=ids,
                )
                set_status(f"saved {path}")
            except Exception as exc:
                log_exception("droplet-wise recording failed", exc)
                set_status("recording failed; see log")
            finally:
                state.is_recording = False

        threading.Thread(target=worker, daemon=True, name="droplet-intensity-recorder").start()

    controls["apply_detector"].on_click(on_apply_detector)
    controls["start_preview"].on_click(on_start)
    controls["stop"].on_click(on_stop)
    controls["snapshot_raw"].on_click(on_snapshot_raw)
    controls["snapshot_overlay"].on_click(on_snapshot_overlay)
    controls["record"].on_click(on_record)
    if "clear_log" in controls:
        controls["clear_log"].on_click(lambda _: setattr(controls["log"], "value", ""))
    return holder
