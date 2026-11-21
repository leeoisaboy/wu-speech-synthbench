"""Render README title banners, WSB dog mascot logo, and pipeline diagrams with Pillow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
MASCOT = ASSETS / "mascot.png"

NAVY = (12, 47, 89)
BLUE = (0, 113, 227)
TEAL = (43, 156, 148)
CORAL = (230, 106, 78)
AMBER = (214, 135, 31)
PURPLE = (123, 107, 176)
MUTED = (88, 105, 122)
GRADIENT = (BLUE, TEAL, AMBER)

FONT_TITLE = Path(r"C:\Windows\Fonts\bahnschrift.ttf")
FONT_UI = Path(r"C:\Windows\Fonts\segoeuib.ttf")
FONT_CN = Path(r"C:\Windows\Fonts\msyhbd.ttc")
FONT_CN_LIGHT = Path(r"C:\Windows\Fonts\msyh.ttc")


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size)


def gradient(size: tuple[int, int], colors: tuple[tuple[int, int, int], ...]) -> Image.Image:
    width, height = size
    stops = np.linspace(0.0, 1.0, len(colors))
    positions = np.linspace(0.0, 1.0, width)
    channels = []
    for channel in range(3):
        values = np.interp(positions, stops, [color[channel] for color in colors])
        channels.append(values.astype(np.uint8))
    row = np.stack(channels, axis=1)[None, :, :].repeat(height, axis=0)
    return Image.fromarray(row, mode="RGB")


def gradient_text(
    canvas: Image.Image,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    colors: tuple[tuple[int, int, int], ...] = GRADIENT,
) -> None:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 12
    height = bottom - top + 12
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).text((6 - left, 6 - top), text, font=text_font, fill=255)
    canvas.paste(gradient(mask.size, colors), (xy[0] - 6, xy[1] - 6), mask)


def fit_font(path: Path, text: str, max_width: int, start: int, minimum: int) -> ImageFont.FreeTypeFont:
    size = start
    while size > minimum:
        candidate = font(path, size)
        if candidate.getbbox(text)[2] <= max_width:
            return candidate
        size -= 2
    return font(path, minimum)


def shadow_card(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    radius: int,
    fill: tuple[int, int, int, int] = (255, 255, 255, 232),
    shadow: int = 12,
) -> None:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    draw.rounded_rectangle((x0 + 3, y0 + shadow, x1 + 3, y1 + shadow), radius=radius, fill=(15, 49, 75, 30))
    overlay = overlay.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(overlay)
    ImageDraw.Draw(canvas).rounded_rectangle(box, radius=radius, fill=fill)


def pill(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    color: tuple[int, int, int],
) -> int:
    left, top, right, bottom = text_font.getbbox(text)
    width = right - left + 34
    height = bottom - top + 18
    draw.rounded_rectangle(
        (xy[0], xy[1], xy[0] + width, xy[1] + height),
        radius=height // 2,
        fill=(255, 255, 255, 238),
        outline=(*color, 185),
        width=2,
    )
    draw.text((xy[0] + 17, xy[1] + 8), text, font=text_font, fill=color)
    return width


def wave(draw: ImageDraw.ImageDraw, origin: tuple[int, int], width: int, color: tuple[int, int, int]) -> None:
    x0, y0 = origin
    points = []
    for index in range(width):
        x = x0 + index
        phase = index / width
        y = y0 - 15 * np.sin(phase * 6.0 * np.pi) * (0.35 + 0.65 * phase)
        points.append((x, y))
    draw.line(points, fill=(*color, 145), width=3, joint="curve")


def mascot_layer(size: int) -> Image.Image:
    image = Image.open(MASCOT).convert("RGBA")
    ratio = size / image.width
    return image.resize((size, int(image.height * ratio)), Image.Resampling.LANCZOS)


def title_banner(language: str) -> None:
    width, height = 1600, 500
    canvas = Image.new("RGBA", (width, height), (247, 250, 254, 255))
    draw = ImageDraw.Draw(canvas)

    for index in range(height):
        blend = index / height
        color = (int(249 - 8 * blend), int(251 - 4 * blend), int(255 - 5 * blend), 255)
        draw.line((0, index, width, index), fill=color)

    draw.rounded_rectangle((15, 15, width - 15, height - 15), radius=34, outline=(218, 226, 240, 255), width=2)
    draw.ellipse((1175, 65, 1550, 440), fill=(*TEAL, 22))
    draw.ellipse((1228, 110, 1518, 400), outline=(*BLUE, 68), width=3)

    dog = mascot_layer(265)
    canvas.alpha_composite(dog, (1240, 92))

    label_font = font(FONT_UI, 21)
    draw.text((82, 48), "PURE NUMPY  /  SYNTHETIC CORPUS  /  HONEST SPEECH BENCHMARK", font=label_font, fill=BLUE)

    title_font = fit_font(FONT_TITLE, "Wu-Speech-SynthBench", 1040, 86, 62)
    gradient_text(canvas, (80, 88), "Wu-Speech-SynthBench", title_font)

    if language == "zh":
        subtitle = "零资源方言语音合成规划 · 信号切字与诚实评测工坊"
        description = "三位置防泄漏语料规划、Parseval 能量谷切字、先验概率校准与匈牙利防刷分评测"
        chips = ("能量谷切字", "先验概率校准", "匈牙利多区间", "泄漏 X 光体检")
        footer = "一只小狗，全仓纯 NumPy，把语音评测里最隐蔽的五个工程坑一次修干净"
    else:
        subtitle = "Zero-resource dialect corpus planning · DSP timestamping & honest evaluation"
        description = "Balanced 3-position manifests, Parseval energy-valley cutting, prior calibration & Hungarian matching"
        chips = ("Energy Valley DSP", "Prior Calibration", "Hungarian IoU", "Leakage X-Ray")
        footer = "One dog, pure NumPy, eliminating the five hidden pitfalls of speech retrieval benchmarks"

    subtitle_font = fit_font(FONT_CN if language == "zh" else FONT_UI, subtitle, 1040, 40, 26)
    draw.text((82, 224), subtitle, font=subtitle_font, fill=NAVY)
    desc_font = fit_font(FONT_CN_LIGHT if language == "zh" else FONT_UI, description, 1040, 24, 18)
    draw.text((84, 286), description, font=desc_font, fill=MUTED)

    chip_font = font(FONT_CN if language == "zh" else FONT_UI, 20)
    cursor = 84
    chip_cols = (BLUE, TEAL, PURPLE, CORAL)
    for index, chip in enumerate(chips):
        cursor += pill(draw, (cursor, 342), chip, chip_font, chip_cols[index]) + 12

    wave(draw, (84, 437), 900, TEAL)
    draw.line((84, 415, 220, 415), fill=(*BLUE, 255), width=5)
    draw.line((220, 415, 440, 415), fill=(*TEAL, 255), width=5)
    draw.line((440, 415, 720, 415), fill=(*AMBER, 255), width=5)

    footer_font = font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 19)
    draw.text((84, 458), footer, font=footer_font, fill=(129, 145, 157))

    canvas.convert("RGB").save(ASSETS / f"title_banner.{language}.png", optimize=True)


def logo() -> None:
    size = 512
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    shadow_card(canvas, (12, 12, 500, 500), radius=112, fill=(252, 253, 255, 255), shadow=10)
    draw = ImageDraw.Draw(canvas)
    draw.ellipse((38, 132, 260, 354), fill=(*TEAL, 24))
    draw.ellipse((45, 139, 253, 347), outline=(*BLUE, 80), width=3)

    dog = mascot_layer(178)
    canvas.alpha_composite(dog, (52, 150))

    gradient_text(canvas, (254, 132), "WSB", font(FONT_TITLE, 114), GRADIENT)
    draw.text((260, 268), "SYNTHBENCH", font=font(FONT_UI, 22), fill=BLUE)
    draw.text((260, 300), "QA STUDIO", font=font(FONT_UI, 22), fill=TEAL)
    draw.text((260, 365), "PURE NUMPY V3", font=font(FONT_UI, 16), fill=MUTED)
    draw.ellipse((260, 401, 273, 414), fill=BLUE)
    draw.ellipse((290, 401, 303, 414), fill=TEAL)
    draw.ellipse((320, 401, 333, 414), fill=AMBER)

    canvas.save(ASSETS / "logo.png", optimize=True)


def pipeline(language: str) -> None:
    width, height = 1600, 900
    canvas = Image.new("RGBA", (width, height), (247, 250, 254, 255))
    draw = ImageDraw.Draw(canvas)
    for index in range(height):
        blend = index / height
        draw.line((0, index, width, index), fill=(int(248 - 6 * blend), int(251 - 4 * blend), int(255 - 4 * blend), 255))

    draw.ellipse((1180, 38, 1545, 230), fill=(*TEAL, 20))
    dog = mascot_layer(135)
    canvas.alpha_composite(dog, (1340, 22))

    title = "一套标定工坊，五个精算关口" if language == "zh" else "One QA factory, five precision gates"
    draw.text((64, 58), title, font=font(FONT_CN if language == "zh" else FONT_TITLE, 54), fill=NAVY)
    subtitle = (
        "从零资源语料规划、Parseval 信号切字到先验校准与匈牙利诚实评测：全仓纯 NumPy、零重型依赖"
        if language == "zh"
        else "From balanced corpus manifests and Parseval DSP cutting to prior calibration & honest Hungarian matching"
    )
    draw.text((68, 135), subtitle, font=fit_font(FONT_CN_LIGHT if language == "zh" else FONT_UI, subtitle, 1100, 23, 17), fill=MUTED)
    draw.line((68, 178, 440, 178), fill=BLUE, width=5)
    draw.line((440, 178, 680, 178), fill=TEAL, width=5)

    if language == "zh":
        cards = (
            ("01", "语料规划与体检", ("三位置防死锁", "未注册硬负样本", "全词表泄漏扫描"), BLUE),
            ("02", "能量谷切字", ("词内局部 Min-Max", "Parseval 高频子带", "帧中心后验对齐"), TEAL),
            ("03", "先验概率校准", ("岭回归 Armijo 搜索", "log-odds 先验修正", "ECE & Brier 标定"), PURPLE),
            ("04", "多词集合解码", ("Robust Median-MAD", "continue 防误杀", "F1 vs FAR 工作面"), AMBER),
            ("05", "诚实定位与排序", ("匈牙利最优匹配", "零重叠与无位惩罚", "头/中/尾词频雷达"), CORAL),
        )
        bullet_font = font(FONT_CN_LIGHT, 20)
        card_title_font = font(FONT_CN, 24)
        footer = "每个关口都有严格单元测试、命令行一键复现与 SynthBench Studio 浏览器交互沙盒"
    else:
        cards = (
            ("01", "Manifest & Audit", ("3-position balance", "disjoint hard negs", "full leakage scan"), BLUE),
            ("02", "Energy Valley DSP", ("word-local Min-Max", "Parseval high-band", "frame-center align"), TEAL),
            ("03", "Prior Calibration", ("ridge Armijo Newton", "log-odds correction", "ECE & Brier metrics"), PURPLE),
            ("04", "Set Decoding", ("Robust Median-MAD", "continue rescue", "F1 vs FAR surface"), AMBER),
            ("05", "Honest Eval", ("Hungarian matching", "zero-IoU & no-pos", "head/body/tail bias"), CORAL),
        )
        bullet_font = font(FONT_UI, 18)
        card_title_font = font(FONT_UI, 23)
        footer = "Every gate is backed by pure-NumPy unit tests, CLI pipelines, and the live SynthBench Studio"

    x_positions = (62, 368, 674, 980, 1286)
    for (number, heading, bullets, color), x in zip(cards, x_positions):
        shadow_card(canvas, (x, 230, x + 252, 652), radius=22)
        draw.rounded_rectangle((x, 230, x + 252, 242), radius=6, fill=color)
        draw.ellipse((x + 22, 265, x + 76, 319), fill=color)
        number_font = font(FONT_UI, 22)
        nw = number_font.getbbox(number)[2]
        draw.text((x + 49 - nw / 2, 280), number, font=number_font, fill=(255, 255, 255))
        draw.text((x + 20, 340), heading, font=card_title_font, fill=NAVY)
        for row, bullet in enumerate(bullets):
            y = 410 + row * 63
            draw.ellipse((x + 24, y + 7, x + 35, y + 18), fill=color)
            bf = fit_font(FONT_CN_LIGHT if language == "zh" else FONT_UI, bullet, 180, bullet_font.size, 14)
            draw.text((x + 48, y), bullet, font=bf, fill=(64, 80, 96))
        draw.line((x + 24, 585, x + 228, 585), fill=(222, 232, 235), width=2)

    for x_conn in (312, 618, 924, 1230):
        draw.line((x_conn, 441, x_conn + 56, 441), fill=(162, 181, 189), width=3)

    shadow_card(canvas, (62, 696, 1538, 834), radius=22, fill=(255, 255, 255, 210), shadow=8)
    sentence = (
        "规划防泄漏，切字保清音，校准归真值，解码抗尖峰，评测拒水分。"
        if language == "zh"
        else "Leak-free manifests, fricative-safe cutting, calibrated posteriors, robust decoding, honest metrics."
    )
    draw.text((94, 724), sentence, font=font(FONT_CN if language == "zh" else FONT_UI, 22), fill=NAVY)
    draw.text((94, 774), footer, font=font(FONT_CN_LIGHT if language == "zh" else FONT_UI, 20), fill=MUTED)
    wave(draw, (92, 806), 1160, TEAL)

    canvas.convert("RGB").save(ASSETS / f"pipeline.{language}.png", optimize=True)


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    title_banner("zh")
    title_banner("en")
    logo()
    pipeline("zh")
    pipeline("en")
    print("assets written to", ASSETS)
