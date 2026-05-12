from __future__ import annotations

from pathlib import Path

try:
    import ipywidgets as widgets
    from IPython.display import display
except Exception:  # pragma: no cover
    widgets = None
    display = None


class WidgetFactoryUnavailable(RuntimeError):
    pass


def require_widgets():
    if widgets is None:
        raise WidgetFactoryUnavailable("ipywidgets is not available")
    return widgets


def make_basic_controls(default_root: str = "experiments"):
    """Create the main notebook controls.

    The controls intentionally expose the parameters that are usually adjusted by
    eye during setup: exposure, FPS, detector percentile/radius/min-distance,
    downsampling, correlation backend, colorized preview, and camera vs video
    replay source.
    """
    w = require_widgets()
    controls = {
        # source / output
        "root": w.Text(value=default_root, description="Root"),
        "source_kind": w.ToggleButtons(options=[("Camera", "camera"), ("Video file", "video")], value="camera", description="Source"),
        "video_path": w.Text(value="", description="Video"),
        "video_loop": w.Checkbox(value=True, description="Loop video"),
        "fake_realtime": w.Checkbox(value=True, description="Replay @ FPS"),
        # camera
        "fps": w.FloatText(value=20.0, description="FPS"),
        "exposure_ms": w.FloatText(value=50.0, description="Exposure ms"),
        "duration_s": w.FloatText(value=60.0, description="Duration s"),
        # detector
        "detector_header": w.HTML(value="<b>Detector parameters</b>"),
        "percentile": w.FloatSlider(value=89.0, min=50.0, max=99.9, step=0.1, description="Percentile", continuous_update=False),
        "downscale_factor": w.IntSlider(value=2, min=1, max=16, step=1, description="Downscale", continuous_update=False),
        "droplet_radius_px": w.IntText(value=100, description="Radius px"),
        "min_distance": w.IntText(value=40, description="Min dist"),
        "corr_backend": w.ToggleButtons(options=["fft", "ndimage"], value="fft", description="Corr"),
        "cpu_downscale": w.Checkbox(value=True, description="CPU downscale"),
        "peak_downscale": w.IntSlider(value=1, min=1, max=8, step=1, description="Peak ds", continuous_update=False),
        "apply_detector": w.Button(description="Apply detector params", button_style="info"),
        # tracker / target
        "favorite_id": w.IntText(value=0, description="Favorite ID"),
        "ids": w.Text(value="", description="IDs"),
        "finish_enabled": w.Checkbox(value=False, description="Finish zone"),
        "finish_x0": w.IntText(value=0, description="x0"),
        "finish_y0": w.IntText(value=0, description="y0"),
        "finish_x1": w.IntText(value=300, description="x1"),
        "finish_y1": w.IntText(value=0, description="y1"),
        "finish_width": w.IntText(value=80, description="zone w"),
        # display / acquisition actions
        "colorize": w.Checkbox(value=True, description="Color preview"),
        "draw_ids": w.Checkbox(value=True, description="Draw IDs"),
        "draw_detections": w.Checkbox(value=False, description="Draw detections"),
        "beep": w.Checkbox(value=True, description="Beep on hit"),
        "reset_tracker": w.Checkbox(value=True, description="Reset tracker on start"),
        "start_preview": w.Button(description="Start preview/stream", button_style="success"),
        "stop": w.Button(description="Stop", button_style="danger"),
        "snapshot_raw": w.Button(description="Save raw frame"),
        "snapshot_overlay": w.Button(description="Save overlay frame"),
        # video recording
        "video_record_header": w.HTML(value="<b>Video recording</b>"),
        "record_raw_video": w.Checkbox(value=False, description="Record raw video"),
        "record_overlay_video": w.Checkbox(value=False, description="Record overlay video"),
        "raw_video_name": w.Text(value="raw_video.avi", description="Raw name"),
        "overlay_video_name": w.Text(value="overlay_video.avi", description="Overlay name"),
        # droplet intensity recording
        "intensity_header": w.HTML(value="<b>Droplet-wise intensity recording</b>"),
        "roi_radius_px": w.IntText(value=80, description="ROI radius"),
        "output_name": w.Text(value="droplet_intensity.tsv", description="TSV output"),
        "record": w.Button(description="Record intensities", button_style="warning"),
        "status": w.HTML(value="<b>Status:</b> idle"),
        "clear_log": w.Button(description="Clear log"),
        "log": w.Textarea(value="", description="Log", layout=w.Layout(width="100%", height="220px")),
    }
    return controls


def display_controls(controls: dict):
    if display is None:
        raise WidgetFactoryUnavailable("IPython display is not available")
    w = require_widgets()
    box = w.VBox([
        w.HBox([controls["root"], controls["source_kind"]]),
        w.HBox([controls["video_path"], controls["video_loop"], controls["fake_realtime"]]),
        w.HBox([controls["fps"], controls["exposure_ms"], controls["duration_s"]]),
        controls["detector_header"],
        w.HBox([controls["percentile"], controls["corr_backend"], controls["cpu_downscale"]]),
        w.HBox([controls["downscale_factor"], controls["peak_downscale"]]),
        w.HBox([controls["droplet_radius_px"], controls["min_distance"], controls["apply_detector"]]),
        w.HBox([controls["colorize"], controls["draw_ids"], controls["draw_detections"], controls["beep"], controls["reset_tracker"]]),
        w.HBox([controls["favorite_id"], controls["ids"], controls["finish_enabled"]]),
        w.HBox([controls["finish_x0"], controls["finish_y0"], controls["finish_x1"], controls["finish_y1"], controls["finish_width"]]),
        w.HBox([controls["start_preview"], controls["stop"], controls["snapshot_raw"], controls["snapshot_overlay"]]),
        controls["video_record_header"],
        w.HBox([controls["record_raw_video"], controls["raw_video_name"], controls["record_overlay_video"], controls["overlay_video_name"]]),
        controls["intensity_header"],
        w.HBox([controls["roi_radius_px"], controls["output_name"], controls["record"]]),
        controls["status"],
        w.HBox([controls["clear_log"]]),
        controls["log"],
    ])
    display(box)
    return box


def parse_ids(text: str) -> list[int]:
    if not text.strip():
        return []
    out = []
    for part in text.replace(";", ",").replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return sorted(set(x for x in out if x > 0))
