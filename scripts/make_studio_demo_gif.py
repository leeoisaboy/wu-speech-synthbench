"""Capture crisp full-module PNG screenshots and animated demo GIF of SynthBench Studio via Playwright."""

from __future__ import annotations

import functools
import http.server
import io
import socketserver
import threading
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
STUDIO_DIR = ROOT / "studio"
ASSETS_DIR = ROOT / "docs" / "assets"


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _start_server() -> tuple[socketserver.TCPServer, int]:
    handler = functools.partial(_QuietHandler, directory=str(STUDIO_DIR))
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def _capture_full_section(page, selector: str, out_path: Path, max_width: int = 1280) -> Image.Image:
    loc = page.locator(selector)
    loc.scroll_into_view_if_needed()
    page.wait_for_timeout(450)
    raw = loc.screenshot(type="png")
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(round(img.height * ratio))), Image.Resampling.LANCZOS)
    img.save(out_path, optimize=True)
    return img


def _fit_frame(img: Image.Image, target_w: int = 1080, target_h: int = 680) -> Image.Image:
    """Letterbox/fit an entire module screenshot cleanly inside a fixed crisp 1080x680 canvas."""
    bg = Image.new("RGB", (target_w, target_h), (251, 251, 253))
    scale = min((target_w - 24) / img.width, (target_h - 24) / img.height)
    nw = max(1, int(round(img.width * scale)))
    nh = max(1, int(round(img.height * scale)))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    bg.paste(resized, ((target_w - nw) // 2, (target_h - nh) // 2))
    return bg


def main() -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    httpd, port = _start_server()
    url = f"http://127.0.0.1:{port}/index.html"
    gif_frames: list[Image.Image] = []
    gif_durations: list[int] = []

    try:
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome", None):
                try:
                    browser = p.chromium.launch(headless=True, channel=channel) if channel else p.chromium.launch(headless=True)
                    break
                except Exception:
                    continue
            if browser is None:
                raise RuntimeError("No Chromium/Edge/Chrome browser available for Playwright")
            context = browser.new_context(
                viewport={"width": 1380, "height": 960},
                device_scale_factor=2,
            )
            page = context.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_selector("body[data-ready='true']", timeout=15000)
            # Ensure all sections are revealed and sticky topbar never overlaps section screenshots
            page.evaluate(
                """() => {
                  document.querySelectorAll('[data-reveal]').forEach(el => el.classList.add('is-inview'));
                  document.documentElement.style.scrollBehavior = 'auto';
                  const tb = document.querySelector('.topbar');
                  if (tb) tb.style.position = 'static';
                }"""
            )
            page.wait_for_timeout(700)

            def grab_section_frames(selector: str, count: int, step_ms: int, dur_ms: int) -> None:
                loc = page.locator(selector)
                loc.scroll_into_view_if_needed()
                page.wait_for_timeout(280)
                for _ in range(count):
                    raw = loc.screenshot(type="png")
                    img = Image.open(io.BytesIO(raw)).convert("RGB")
                    gif_frames.append(_fit_frame(img))
                    gif_durations.append(dur_ms)
                    page.wait_for_timeout(step_ms)

            # 1. Module 01: DSP (all 3 tracks visible + playhead sweeping)
            _capture_full_section(page, "#sec-dsp", ASSETS_DIR / "demo-dsp.png")
            grab_section_frames("#sec-dsp", count=7, step_ms=160, dur_ms=220)
            # Toggle local norm off briefly to show contrast, then back on
            page.locator("#toggle-local-norm").click()
            page.wait_for_timeout(220)
            grab_section_frames("#sec-dsp", count=4, step_ms=160, dur_ms=260)
            page.locator("#toggle-local-norm").click()
            page.wait_for_timeout(220)

            # 2. Module 02: Decoder & Reliability Calibration
            _capture_full_section(page, "#sec-calib", ASSETS_DIR / "demo-calib.png")
            grab_section_frames("#sec-calib", count=6, step_ms=160, dur_ms=220)
            # Toggle prior fix off then on to show Reliability Curve snap
            page.locator("#toggle-prior-fix").click()
            page.wait_for_timeout(240)
            grab_section_frames("#sec-calib", count=4, step_ms=160, dur_ms=260)
            page.locator("#toggle-prior-fix").click()
            page.wait_for_timeout(240)
            grab_section_frames("#sec-calib", count=3, step_ms=160, dur_ms=240)

            # 3. Module 03: Hungarian Multi-Span & Frequency Bias
            _capture_full_section(page, "#sec-eval", ASSETS_DIR / "demo-eval.png")
            grab_section_frames("#sec-eval", count=6, step_ms=160, dur_ms=220)
            page.locator("#btn-drift-demo").click()
            page.wait_for_timeout(220)
            grab_section_frames("#sec-eval", count=4, step_ms=160, dur_ms=260)
            page.locator("#btn-reset-spans").click()
            page.wait_for_timeout(180)

            # 4. Module 04: Manifest Sandbox & Leakage X-Ray
            _capture_full_section(page, "#sec-plan", ASSETS_DIR / "demo-plan.png")
            grab_section_frames("#sec-plan", count=6, step_ms=160, dur_ms=220)
            page.locator("#btn-plan-leak").click()
            page.wait_for_timeout(260)
            grab_section_frames("#sec-plan", count=5, step_ms=160, dur_ms=260)

            browser.close()
    finally:
        httpd.shutdown()

    # Quantize with adaptive palette for ultra-sharp text and clean colors
    paletted = [f.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in gif_frames]
    out_gif = ASSETS_DIR / "studio-demo.gif"
    paletted[0].save(
        out_gif,
        save_all=True,
        append_images=paletted[1:],
        duration=gif_durations,
        loop=0,
        optimize=True,
    )
    print(f"Saved {len(paletted)} crisp full-module frames to {out_gif} ({out_gif.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
