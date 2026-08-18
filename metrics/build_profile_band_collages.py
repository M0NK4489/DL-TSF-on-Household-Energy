from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "profile_band_collages"
PLOT_NAME = "01_multi_window_profile_bands.png"

FREQUENCIES = ("30min", "1H", "2H", "3H")
FREQUENCY_LABELS = {
    "30min": "30 min",
    "1H": "1 h",
    "2H": "2 h",
    "3H": "3 h",
}
HORIZONS = ("24h", "7d", "28d")
HORIZON_LABELS = {"24h": "24 h", "7d": "7 d", "28d": "28 d"}
STEPS_PER_DAY = {"30min": 48, "1H": 24, "2H": 12, "3H": 8}


@dataclass(frozen=True)
class ModelSpec:
    title: str
    filename: str
    scope: str
    prefix: str
    selection: str


MODELS = (
    ModelSpec(
        "Context RNN Diffusion",
        "context_rnn_diffusion_profile_bands.png",
        "Context Forecasting",
        "forecast-rnn-diffusion",
        "context",
    ),
    ModelSpec(
        "Context Transformer Diffusion",
        "context_transformer_diffusion_profile_bands.png",
        "Context Forecasting",
        "forecast-transformer-diffusion",
        "context",
    ),
    ModelSpec(
        "Context RNN VAE",
        "context_rnn_vae_profile_bands.png",
        "Context Forecasting",
        "forecast-rnn-vae",
        "context",
    ),
    ModelSpec(
        "Context Transformer VAE",
        "context_transformer_vae_profile_bands.png",
        "Context Forecasting",
        "forecast-transformer-vae",
        "context",
    ),
    ModelSpec(
        "No-context RNN Diffusion",
        "no_context_rnn_diffusion_profile_bands.png",
        "No Context Forecasting",
        "forecast-nocontext-rnn-diffusion",
        "no_context",
    ),
    ModelSpec(
        "No-context Transformer Diffusion",
        "no_context_transformer_diffusion_profile_bands.png",
        "No Context Forecasting",
        "forecast-nocontext-transformer-diffusion",
        "no_context",
    ),
    ModelSpec(
        "No-context RNN VAE",
        "no_context_rnn_vae_profile_bands.png",
        "No Context Forecasting",
        "forecast-nocontext-rnn-vae",
        "no_context",
    ),
    ModelSpec(
        "No-context Transformer VAE",
        "no_context_transformer_vae_profile_bands.png",
        "No Context Forecasting",
        "forecast-nocontext-transformer-vae",
        "no_context",
    ),
    ModelSpec(
        "iTransformer Baseline",
        "baseline_itransformer_profile_bands.png",
        "Context Forecasting",
        "baseline-itransformer",
        "baseline",
    ),
    ModelSpec(
        "N-HiTS Baseline",
        "baseline_nhits_profile_bands.png",
        "Context Forecasting",
        "baseline-nhits",
        "baseline",
    ),
    ModelSpec(
        "Temporal Fusion Transformer Baseline",
        "baseline_tft_profile_bands.png",
        "Context Forecasting",
        "baseline-tft",
        "baseline",
    ),
    ModelSpec(
        "TSMixer Baseline",
        "baseline_tsmixer_profile_bands.png",
        "Context Forecasting",
        "baseline-tsmixer",
        "baseline",
    ),
)


def load_font(filename: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
    if font_path.exists():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


TITLE_FONT = load_font("arialbd.ttf", 44)
LABEL_FONT = load_font("arial.ttf", 32)
AXIS_FONT = load_font("arialbd.ttf", 34)


def expected_context_steps(horizon: str, frequency: str) -> int:
    days = 2 if horizon == "24h" else 7
    return days * STEPS_PER_DAY[frequency]


def find_plot(spec: ModelSpec, horizon: str, frequency: str) -> Path:
    scope_dir = ROOT / spec.scope
    prefix = f"{spec.prefix}__freq-{frequency}__hor-{horizon}__"
    expected_context = expected_context_steps(horizon, frequency)
    matches: list[Path] = []

    for run_dir in scope_dir.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(prefix):
            continue

        if spec.selection == "context":
            selected = f"__ctx-{expected_context}__" in run_dir.name
        elif spec.selection == "no_context":
            selected = "__ctx-0__" in run_dir.name
        else:
            selected = (
                f"__in-{expected_context}__" in run_dir.name
                and "__q-10-50-90" in run_dir.name
            )

        plot_path = run_dir / "plots" / PLOT_NAME
        if selected and plot_path.exists():
            matches.append(plot_path)

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one plot for {spec.title}, {horizon}, {frequency}; "
            f"found {len(matches)}."
        )
    return matches[0]


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    width: int = 4,
) -> None:
    draw.line((start, end), fill="black", width=width)
    if start[1] == end[1]:
        draw.polygon(
            ((end[0], end[1]), (end[0] - 18, end[1] - 10), (end[0] - 18, end[1] + 10)),
            fill="black",
        )
    else:
        draw.polygon(
            ((end[0], end[1]), (end[0] - 10, end[1] + 18), (end[0] + 10, end[1] + 18)),
            fill="black",
        )


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    draw.text(xy, text, fill="black", font=font, anchor="mm")


def draw_vertical_text(
    canvas: Image.Image,
    center: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    probe = ImageDraw.Draw(canvas)
    bbox = probe.textbbox((0, 0), text, font=font)
    text_image = Image.new("RGBA", (bbox[2] - bbox[0] + 20, bbox[3] - bbox[1] + 20), "white")
    text_draw = ImageDraw.Draw(text_image)
    text_draw.text((10 - bbox[0], 10 - bbox[1]), text, fill="black", font=font)
    rotated = text_image.rotate(90, expand=True, fillcolor="white")
    position = (center[0] - rotated.width // 2, center[1] - rotated.height // 2)
    canvas.paste(rotated, position, rotated)


def build_collage(spec: ModelSpec) -> Path:
    panel_width = 1400
    panel_height = 491
    column_gap = 24
    row_gap = 24
    left_margin = 230
    right_margin = 45
    title_height = 82
    column_label_height = 52
    bottom_margin = 135

    grid_width = len(FREQUENCIES) * panel_width + (len(FREQUENCIES) - 1) * column_gap
    grid_height = len(HORIZONS) * panel_height + (len(HORIZONS) - 1) * row_gap
    canvas_width = left_margin + grid_width + right_margin
    grid_top = title_height + column_label_height
    canvas_height = grid_top + grid_height + bottom_margin

    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw_centered_text(draw, (canvas_width // 2, 36), spec.title, TITLE_FONT)

    for column, frequency in enumerate(FREQUENCIES):
        x = left_margin + column * (panel_width + column_gap)
        draw_centered_text(
            draw,
            (x + panel_width // 2, title_height + column_label_height // 2),
            FREQUENCY_LABELS[frequency],
            LABEL_FONT,
        )

    for row, horizon in enumerate(HORIZONS):
        y = grid_top + row * (panel_height + row_gap)
        draw_centered_text(
            draw,
            (left_margin - 55, y + panel_height // 2),
            HORIZON_LABELS[horizon],
            LABEL_FONT,
        )

        for column, frequency in enumerate(FREQUENCIES):
            x = left_margin + column * (panel_width + column_gap)
            plot_path = find_plot(spec, horizon, frequency)
            with Image.open(plot_path) as source:
                panel = ImageOps.contain(
                    source.convert("RGB"),
                    (panel_width, panel_height),
                    Image.Resampling.LANCZOS,
                )
            panel_x = x + (panel_width - panel.width) // 2
            panel_y = y + (panel_height - panel.height) // 2
            canvas.paste(panel, (panel_x, panel_y))

    axis_x = 72
    axis_y = grid_top + grid_height + 48
    draw_arrow(draw, (axis_x, axis_y), (canvas_width - 25, axis_y))
    draw_arrow(draw, (axis_x, axis_y), (axis_x, grid_top - 8))
    draw_centered_text(
        draw,
        (left_margin + grid_width // 2, canvas_height - 34),
        "Sampling frequency",
        AXIS_FONT,
    )
    draw_vertical_text(
        canvas,
        (25, grid_top + grid_height // 2),
        "Forecast horizon",
        AXIS_FONT,
    )

    output_path = OUTPUT_DIR / spec.filename
    canvas.save(output_path, "PNG", optimize=True, dpi=(300, 300))
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MODELS:
        output_path = build_collage(spec)
        print(output_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
