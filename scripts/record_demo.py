#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from calibench import audit

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SIZE = 18
LINE_HEIGHT = 26
WIDTH = 960
HEIGHT = 540
BG = (18, 18, 18)
FG = (204, 204, 204)
GREEN = (80, 200, 120)
YELLOW = (230, 190, 80)
CYAN = (100, 200, 220)
MAGENTA = (180, 120, 220)
PROMPT_COLOUR = GREEN
FPS = 4
HOLD_FRAMES = 8
PADDING_X = 24
PADDING_Y = 20

OUTPUT = Path(__file__).resolve().parent.parent / "assets" / "demo.mp4"

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
font_bold = ImageFont.truetype(FONT_BOLD_PATH, FONT_SIZE)


def blank() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), BG)


def draw_lines(
    img: Image.Image,
    lines: list[tuple[str, tuple[int, int, int]]],
    start_y: int = PADDING_Y,
) -> Image.Image:
    draw = ImageDraw.Draw(img)
    y = start_y
    for text, colour in lines:
        draw.text((PADDING_X, y), text, fill=colour, font=font)
        y += LINE_HEIGHT
    return img


def prompt_line(text: str) -> tuple[str, tuple[int, int, int]]:
    return (f">>> {text}", PROMPT_COLOUR)


def output_line(text: str) -> tuple[str, tuple[int, int, int]]:
    return (text, FG)


def heading_line(text: str) -> tuple[str, tuple[int, int, int]]:
    return (text, YELLOW)


def build_typing_frames(
    prefix_lines: list[tuple[str, tuple[int, int, int]]],
    code_text: str,
) -> list[Image.Image]:
    frames = []
    full = f">>> {code_text}"
    for i in range(4, len(full) + 1, 2):
        img = blank()
        current_lines = prefix_lines + [(full[:i] + "_", PROMPT_COLOUR)]
        draw_lines(img, current_lines)
        frames.append(img)
    img = blank()
    draw_lines(img, prefix_lines + [(full, PROMPT_COLOUR)])
    frames.append(img)
    return frames


def build_plot_frame(report) -> Image.Image:
    fig = report.plot()
    fig.set_size_inches(WIDTH / 100, HEIGHT / 100)
    fig.set_dpi(100)
    fig.patch.set_facecolor("#121212")
    for ax in fig.axes:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#cccccc")
        ax.xaxis.label.set_color("#cccccc")
        ax.yaxis.label.set_color("#cccccc")
        ax.title.set_color("#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#555555")
    fig.suptitle(fig._suptitle.get_text(), color="#e0e0e0", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        fig.savefig(tmp.name, dpi=100, facecolor="#121212")
        plt.close(fig)
        plot_img = Image.open(tmp.name).convert("RGB")

    return plot_img.resize((WIDTH, HEIGHT), Image.LANCZOS)


def main() -> None:
    rng = np.random.default_rng(42)
    y_pred = rng.uniform(0, 10, size=5000)
    true_std = rng.uniform(0.5, 2.0, size=5000)
    y_std = true_std * 0.4
    y_true = y_pred + rng.normal(0.0, true_std)

    report = audit(y_true, y_pred, y_std)
    report_str = str(report)
    report_lines = report_str.split("\n")

    frames: list[Image.Image] = []

    title_img = blank()
    draw_lines(title_img, [
        heading_line("calibench"),
        output_line(""),
        output_line("One-line uncertainty audit for ML models"),
    ], start_y=HEIGHT // 3 - LINE_HEIGHT)
    frames.extend([title_img] * HOLD_FRAMES)

    scene1: list[tuple[str, tuple[int, int, int]]] = []
    import_lines = [
        "import numpy as np",
        "from calibench import audit",
    ]
    for line in import_lines:
        frames.extend(build_typing_frames(scene1, line))
        scene1.append(prompt_line(line))
        img = blank()
        draw_lines(img, scene1)
        frames.extend([img] * 2)

    scene1.append(output_line(""))

    data_lines = [
        "y_true, y_pred, y_std = load_predictions()",
    ]
    for line in data_lines:
        frames.extend(build_typing_frames(scene1, line))
        scene1.append(prompt_line(line))
        img = blank()
        draw_lines(img, scene1)
        frames.extend([img] * 2)

    scene1.append(output_line(""))

    frames.extend(build_typing_frames(scene1, "report = audit(y_true, y_pred, y_std)"))
    scene1.append(prompt_line("report = audit(y_true, y_pred, y_std)"))
    img = blank()
    draw_lines(img, scene1)
    frames.extend([img] * HOLD_FRAMES)

    scene2: list[tuple[str, tuple[int, int, int]]] = []
    frames.extend(build_typing_frames(scene2, "print(report)"))
    scene2.append(prompt_line("print(report)"))
    img = blank()
    draw_lines(img, scene2)
    frames.extend([img] * 2)

    for rline in report_lines:
        scene2.append(output_line(rline))
        img = blank()
        draw_lines(img, scene2)
        frames.append(img)
    img = blank()
    draw_lines(img, scene2)
    frames.extend([img] * HOLD_FRAMES * 2)

    scene3: list[tuple[str, tuple[int, int, int]]] = []
    frames.extend(build_typing_frames(scene3, "report.plot()"))
    scene3.append(prompt_line("report.plot()"))
    img = blank()
    draw_lines(img, scene3)
    frames.extend([img] * 2)

    plot_frame = build_plot_frame(report)
    frames.extend([plot_frame] * HOLD_FRAMES * 2)

    scene4: list[tuple[str, tuple[int, int, int]]] = []
    frames.extend(build_typing_frames(scene4, "report.to_dict()"))
    scene4.append(prompt_line("report.to_dict()"))
    img = blank()
    draw_lines(img, scene4)
    frames.extend([img] * 2)

    d = report.to_dict()
    dict_preview = [
        "{",
        f'  "verdict": "{d["verdict"]}",',
        f'  "temperature": {d["temperature"]:.2f},',
        f'  "ece": {d["ece"]:.4f},',
        f'  "recalibrated_ece": {d["recalibrated_ece"]:.4f},',
        f'  "coverage_at_90": {d["coverage_at_90"]:.4f},',
        f'  "recalibrated_coverage_at_90": {d["recalibrated_coverage_at_90"]:.4f},',
        "  ...",
        "}",
    ]
    for dline in dict_preview:
        scene4.append(output_line(dline))
        img = blank()
        draw_lines(img, scene4)
        frames.append(img)
    img = blank()
    draw_lines(img, scene4)
    frames.extend([img] * HOLD_FRAMES)

    scene5: list[tuple[str, tuple[int, int, int]]] = []
    frames.extend(build_typing_frames(scene5, "calibrated = report.calibrator.transform(new_std)"))
    scene5.append(prompt_line("calibrated = report.calibrator.transform(new_std)"))
    img = blank()
    draw_lines(img, scene5)
    frames.extend([img] * HOLD_FRAMES)

    end_img = blank()
    draw_lines(end_img, [
        heading_line("pip install calibench"),
        output_line(""),
        output_line("github.com/michaelmillar/calibench"),
    ], start_y=HEIGHT // 3 - LINE_HEIGHT)
    frames.extend([end_img] * HOLD_FRAMES)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, frame in enumerate(frames):
            frame.save(Path(tmpdir) / f"frame_{i:05d}.png")

        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", str(FPS),
            "-i", str(Path(tmpdir) / "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(OUTPUT),
        ], check=True)

    print(f"Demo video saved to {OUTPUT}")
    print(f"Frames: {len(frames)}, Duration: {len(frames)/FPS:.1f}s")


if __name__ == "__main__":
    main()
