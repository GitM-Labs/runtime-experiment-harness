"""Startup banner for runtime.

Usage:
    from banner import banner
    print(banner())                 # auto-detects colour support
    print(banner("runtime", subtitle="v0.4.1"))

Run directly to preview:  python banner.py
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence

# --- 5x5 block font -------------------------------------------------------
# '#' = filled cell, '.' = empty. Add glyphs here to support more names.

FONT: dict[str, tuple[str, ...]] = {
    "R": ("####.", "#...#", "####.", "#..#.", "#...#"),
    "U": ("#...#", "#...#", "#...#", "#...#", ".###."),
    "N": ("#...#", "##..#", "#.#.#", "#..##", "#...#"),
    "T": ("#####", "..#..", "..#..", "..#..", "..#.."),
    "I": ("#####", "..#..", "..#..", "..#..", "#####"),
    "M": ("#...#", "##.##", "#.#.#", "#...#", "#...#"),
    "E": ("#####", "#....", "####.", "#....", "#####"),
    "A": (".###.", "#...#", "#####", "#...#", "#...#"),
    "C": (".####", "#....", "#....", "#....", ".####"),
    "D": ("####.", "#...#", "#...#", "#...#", "####."),
    "O": (".###.", "#...#", "#...#", "#...#", ".###."),
    "S": (".####", "#....", ".###.", "....#", "####."),
    "L": ("#....", "#....", "#....", "#....", "#####"),
    "P": ("####.", "#...#", "####.", "#....", "#...."),
    "Y": ("#...#", ".#.#.", "..#..", "..#..", "..#.."),
    "X": ("#...#", ".#.#.", "..#..", ".#.#.", "#...#"),
    "H": ("#...#", "#...#", "#####", "#...#", "#...#"),
    "V": ("#...#", "#...#", "#...#", ".#.#.", "..#.."),
    "G": (".####", "#....", "#..##", "#...#", ".###."),
    "B": ("####.", "#...#", "####.", "#...#", "####."),
    "F": ("#####", "#....", "####.", "#....", "#...."),
    " ": (".....", ".....", ".....", ".....", "....."),
}

BLOCK = "\u25a6"  # ▦
HEIGHT = 5

# Vertical gradient, top row brightest. Truecolor RGB.
GRADIENT = [(0xE6, 0xF7, 0xFF), (0x9A, 0xD9, 0xF5),
            (0x5C, 0xB3, 0xE8), (0x2E, 0x86, 0xC8), (0x1B, 0x5A, 0x9E)]

DIM = "\u001b[2m"
RESET = "\u001b[0m"


def supports_color(stream=None) -> bool:
    """True when it is safe to emit ANSI escapes."""
    stream = stream or sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    if not hasattr(stream, "isatty") or not stream.isatty():
        return False
    return os.environ.get("TERM", "") != "dumb"


def _rows(text: str, cell: str, gap: int) -> list[str]:
    glyphs = []
    for ch in text.upper():
        try:
            glyphs.append(FONT[ch])
        except KeyError:
            raise ValueError(f"no glyph for {ch!r}; add one to FONT") from None

    spacer = " " * gap
    rows = []
    for y in range(HEIGHT):
        parts = ["".join(cell if c == "#" else " " * len(cell) for c in g[y])
                 for g in glyphs]
        rows.append(spacer.join(parts).rstrip())
    return rows


def banner(
    text: str = "runtime",
    subtitle: str | Sequence[str] | None = None,
    *,
    color: bool | None = None,
    cell: str = BLOCK * 2,
    gap: int = 1,
    indent: int = 2,
) -> str:
    """Render `text` as a block banner.

    color=None auto-detects; pass True/False to force.
    cell is the glyph used per filled pixel (two blocks keeps the
    aspect ratio close to square in most terminals).
    """
    rows = _rows(text, cell, gap)
    pad = " " * indent
    use_color = supports_color() if color is None else color

    out = []
    for i, row in enumerate(rows):
        if use_color:
            r, g, b = GRADIENT[i % len(GRADIENT)]
            out.append(f"{pad}\u001b[38;2;{r};{g};{b}m{row}{RESET}")
        else:
            out.append(pad + row)

    if subtitle:
        lines = [subtitle] if isinstance(subtitle, str) else list(subtitle)
        width = max(len(r) for r in rows)
        out.append("")
        for line in lines:
            centered = line.center(width).rstrip()
            out.append(f"{pad}{DIM}{centered}{RESET}" if use_color else pad + centered)

    return "\n".join(out)


def harness_banner(
    version: str = "",
    model: str | None = None,
    *,
    color: bool | None = None,
) -> str:
    """The artwork shown when the harness starts.

    Renders REX in the block font over a dim strapline. `model` is included only
    when known, so `rex check` (which has no manifest) stays uncluttered.
    """
    strapline = "runtime experiment harness"
    if version:
        strapline += f"  \u00b7  v{version}"
        # strapline += "vLLM  \u00b7  guidellm  \u00b7  NVLink fabric"

    subtitles = [strapline]
    if model:
        subtitles.append(model)

    return banner("RUNTIME", subtitle=subtitles, color=color) 


if __name__ == "__main__":
    print()
    print(harness_banner("0.2.0", color=True))
    print()