import pytest

from runtime_harness import __version__
from runtime_harness.banner import FONT, banner, harness_banner
from runtime_harness.experiments import render_runtime_banner


def test_render_runtime_banner_names_the_harness():
    art = render_runtime_banner()
    assert "runtime experiment harness" in art
    assert __version__ in art


def test_banner_renders_block_art():
    art = harness_banner("0.2.0", color=False)
    assert "▦" in art, "expected block glyphs in the artwork"


def test_banner_fits_a_standard_terminal():
    widest = max(len(line) for line in harness_banner("0.2.0", color=False).splitlines())
    assert widest <= 80, f"artwork is {widest} columns; wraps an 80-column terminal"


def test_banner_includes_model_when_known():
    assert "Qwen/Qwen3-8B" in harness_banner("0.2.0", "Qwen/Qwen3-8B", color=False)


def test_banner_omits_model_when_absent():
    art = harness_banner("0.2.0", color=False)
    assert art.count("\n\n") == 1, "expected a single blank line before the strapline"


def test_banner_colour_can_be_forced_off():
    assert "[" not in harness_banner("0.2.0", color=False)


def test_banner_colour_can_be_forced_on():
    assert "[38;2;" in harness_banner("0.2.0", color=True)


def test_font_covers_the_wordmark():
    assert set("RUNTIME") <= set(FONT)


def test_unknown_glyph_is_reported_clearly():
    with pytest.raises(ValueError, match="no glyph for"):
        banner("REX!", color=False)
