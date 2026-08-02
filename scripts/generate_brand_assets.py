"""
Generate every brand raster from ONE source: the Apex mark.

WHY A SCRIPT RATHER THAN HAND-MADE FILES. The previous og-share-card.png was
drawn by hand and DRIFTED from the site logo -- the share preview and the
header stopped matching, and nobody noticed until the card was regenerated.
Deriving every asset from the same geometry means they cannot diverge again:
change the mark here, rerun, and the favicon, app icons and share card all
follow.

Outputs:
    docs/favicon-16.png  favicon-32.png  icon-192.png  icon-512.png
    docs/apple-touch-icon.png   (180, with padding -- iOS crops to a squircle)
    docs/favicon.ico            (16+32 bundled)
    docs/og-share-card.png      (1200x630, the iMessage/Twitter preview)

Usage:
    python3 scripts/generate_brand_assets.py
"""

import math
import os
import subprocess
import tempfile

INK = "#0a0c10"
GOLD = "#d4af37"
WHITE = "#ffffff"

# Outer octagon WHITE, inner apex GOLD -- the mark carrying the same split as
# the wordmark (Octane white, Alpha gold). The earlier two-gold version
# separated at only 1.49:1, so at 16px the octagon and the apex merged into a
# single blurred shape; white against gold is 2.10:1.
# On a LIGHT background white disappears, so the outer stroke takes the ink
# colour instead -- same mark, inverted where it has to be.

# Exact regular octagon, vertices at 22.5 + k*45 degrees.
def octagon_points(cx, cy, r):
    return [(cx + r * math.cos(math.radians(22.5 + k * 45)),
             cy + r * math.sin(math.radians(22.5 + k * 45))) for k in range(8)]


def apex_svg(size, stroke, pad_ratio=0.0, bg=None, on_light=False):
    """
    The mark at a given pixel size.

    Stroke weight is passed in rather than scaled with the artwork: a hairline
    that reads as refined at 512px vanishes at 16px, so small sizes need a
    proportionally heavier line.
    """
    pad = size * pad_ratio
    inner = size - pad * 2
    sc = inner / 100.0
    def P(x, y):
        return f"{pad + x * sc:.2f} {pad + y * sc:.2f}"
    p = octagon_points(50, 50, 40)
    top = (f"M{P(p[3][0], p[3][1])} L{P(p[4][0], p[4][1])} L{P(p[5][0], p[5][1])} "
           f"L{P(p[6][0], p[6][1])} L{P(p[7][0], p[7][1])} L{P(p[0][0], p[0][1])}")
    outer = INK if on_light else WHITE
    rect = f'<rect width="{size}" height="{size}" fill="{bg}"/>' if bg else ""
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}" fill="none">
{rect}
<path d="{top}" stroke="{outer}" stroke-width="{stroke * sc:.2f}" stroke-linecap="round" stroke-linejoin="round"/>
<path d="M{P(32,74)} L{P(50,38)} L{P(68,74)}" stroke="{GOLD}" stroke-width="{stroke * sc:.2f}" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''


def optimize(path):
    """
    Shrink the PNG without touching how it looks.

    wkhtmltoimage writes full 32-bit output, so a 512px two-colour mark came
    out at 1MB and the share card at 3MB. Link-preview crawlers are size
    sensitive and a favicon has no business being that large, but the artwork
    is flat colour -- a small palette is lossless here in practice.
    """
    try:
        from PIL import Image
        im = Image.open(path)
        before = os.path.getsize(path)
        if im.mode in ("RGBA", "LA"):
            # Keep alpha; quantize the colour channels only.
            im = im.quantize(colors=64, method=Image.Quantize.FASTOCTREE)
        else:
            im = im.convert("P", palette=Image.ADAPTIVE, colors=64)
        im.save(path, optimize=True)
        after = os.path.getsize(path)
        return before, after
    except Exception:
        return None, None


def render(svg, out, w, h):
    """SVG -> PNG via wkhtmltoimage, which is already used elsewhere here."""
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(f'<html><body style="margin:0;background:transparent">{svg}</body></html>')
        tmp = f.name
    subprocess.run(["wkhtmltoimage", "--enable-local-file-access", "--transparent",
                    "--width", str(w), "--height", str(h), "--quality", "100",
                    tmp, out], capture_output=True)
    os.unlink(tmp)
    return os.path.exists(out)


def share_card(out):
    """
    1200x630 preview for iMessage, Iessage/Twitter/Slack unfurls.

    Deliberately sparse: at the size a link preview actually renders, a
    crowded card turns to mush. Mark, wordmark, one line.

    No top rule. A 4px gold strip along the edge read as a stray artifact in
    an iMessage bubble rather than as a deliberate accent -- the preview is
    already clipped and rounded by the client, so any edge treatment competes
    with the frame the OS draws around it.
    """
    mark = apex_svg(180, 4.6)
    inner = mark.split(">", 1)[1].rsplit("</svg>", 1)[0]
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
<rect width="1200" height="630" fill="{INK}"/>
<g transform="translate(510,150)">{inner}</g>
<text x="600" y="420" text-anchor="middle" fill="#ffffff"
      font-family="-apple-system, SF Pro Display, Helvetica, Arial, sans-serif"
      font-size="58" font-weight="600" letter-spacing="8">OCTANE <tspan fill="{GOLD}">ALPHA</tspan></text>
<text x="600" y="480" text-anchor="middle" fill="#8b93a3"
      font-family="-apple-system, SF Pro Text, Helvetica, Arial, sans-serif"
      font-size="25" letter-spacing="1.5">Model probability vs. live sportsbook lines</text>
</svg>'''
    return render(svg, out, 1200, 630)


def main():
    os.makedirs("docs", exist_ok=True)
    # (filename, px, stroke, padding, background)
    # Stroke rises as size falls; padding only on the iOS icon, which gets
    # cropped to a squircle and would otherwise clip the mark's corners.
    jobs = [
        ("docs/favicon-16.png", 16, 8.0, 0.0, None),
        ("docs/favicon-32.png", 32, 7.0, 0.0, None),
        ("docs/icon-192.png", 192, 5.2, 0.0, None),
        ("docs/icon-512.png", 512, 4.6, 0.0, None),
        ("docs/apple-touch-icon.png", 180, 5.2, 0.16, INK),
    ]
    ok = 0
    for path, size, stroke, pad, bg in jobs:
        if render(apex_svg(size, stroke, pad, bg), path, size, size):
            ok += 1
            b, a = optimize(path)
            saved = f" [{b//1024}kb -> {a//1024}kb]" if b and a else ""
            print(f"  wrote {path} ({size}px, stroke {stroke}){saved}")
        else:
            print(f"  FAILED {path}")

    # Light-background variant. Not linked from the page today, but the
    # moment the mark lands on anything white the all-white outer vanishes,
    # and generating it here keeps it from being redrawn by hand later --
    # which is exactly how the old share card drifted.
    if render(apex_svg(512, 4.6, on_light=True), "docs/icon-512-light.png", 512, 512):
        optimize("docs/icon-512-light.png")
        ok += 1
        print("  wrote docs/icon-512-light.png (dark outer, for light backgrounds)")

    if share_card("docs/og-share-card.png"):
        ok += 1
        b, a = optimize("docs/og-share-card.png")
        saved = f" [{b//1024}kb -> {a//1024}kb]" if b and a else ""
        print(f"  wrote docs/og-share-card.png (1200x630){saved}")

    # --- SVG favicon: the real fix for desktop rendering ---
    # Chrome prefers an SVG icon when offered and rasterises it itself at
    # whatever size the UI needs, so there is no small-bitmap artifact to go
    # wrong. The .ico stays as a fallback for older browsers.
    with open("docs/favicon.svg", "w", encoding="utf-8") as f:
        f.write(apex_svg(64, 5.6))
    ok += 1
    print("  wrote docs/favicon.svg (vector -- desktop Chrome prefers this)")

    try:
        from PIL import Image
        # Render fresh, UNOPTIMIZED sources for the ICO. Bundling the
        # palette-quantized PNGs produced the stray marks above the icon in
        # desktop Chrome; a clean RGBA source avoids it.
        tmp16, tmp32 = "docs/_ico16.png", "docs/_ico32.png"
        render(apex_svg(16, 8.0), tmp16, 16, 16)
        render(apex_svg(32, 7.0), tmp32, 32, 32)
        base = Image.open(tmp32).convert("RGBA")
        base.save("docs/favicon.ico", format="ICO", sizes=[(16, 16), (32, 32)])
        for t in (tmp16, tmp32):
            if os.path.exists(t):
                os.unlink(t)
        ok += 1
        print("  wrote docs/favicon.ico (from clean RGBA, not the palette PNGs)")
    except Exception as e:
        print(f"  favicon.ico skipped ({e})")

    print(f"\n{ok} asset(s) generated from one source geometry.")


if __name__ == "__main__":
    main()
