"""
A plain geometric "dummy" silhouette (circle head, trapezoid torso, capsule
arms/legs) with three numbered badges overlaid at head/body/leg positions,
for the post-fight Damage Taken section. Deliberately generic and NOT
photorealistic or based on any real person's likeness -- built the same
way as the site's other custom icons: simple primitives with coordinates
that can be verified by reading the numbers, not a traced illustration.

mirror=True flips the whole figure horizontally (badges included) so the
two fighters in a side-by-side comparison can face inward toward each
other, matching how the reference layout reads.
"""


def build_damage_silhouette_svg(head: int, body: int, leg: int, mirror: bool = False, color: str = "#8a8f9a") -> str:
    transform = ' transform="scale(-1,1) translate(-100,0)"' if mirror else ""
    return f"""<svg viewBox="0 0 100 232" width="100%" height="100%" class="damage-silhouette">
  <g{transform}>
    <circle cx="50" cy="22" r="16" fill="none" stroke="{color}" stroke-width="2.5"/>
    <rect x="44" y="36" width="12" height="8" fill="none" stroke="{color}" stroke-width="2.5"/>
    <path d="M30,44 L70,44 L64,112 L36,112 Z" fill="none" stroke="{color}" stroke-width="2.5" stroke-linejoin="round"/>
    <rect x="14" y="46" width="14" height="65" rx="7" fill="none" stroke="{color}" stroke-width="2.5"/>
    <rect x="72" y="46" width="14" height="65" rx="7" fill="none" stroke="{color}" stroke-width="2.5"/>
    <rect x="34" y="112" width="16" height="103" rx="8" fill="none" stroke="{color}" stroke-width="2.5"/>
    <rect x="50" y="112" width="16" height="103" rx="8" fill="none" stroke="{color}" stroke-width="2.5"/>
    <circle cx="20" cy="18" r="15" fill="#ff5c5c"/>
    <text x="20" y="23" text-anchor="middle" class="damage-badge-text">{head}</text>
    <circle cx="18" cy="88" r="15" fill="#ff5c5c"/>
    <text x="18" y="93" text-anchor="middle" class="damage-badge-text">{body}</text>
    <circle cx="16" cy="150" r="15" fill="#ff5c5c"/>
    <text x="16" y="155" text-anchor="middle" class="damage-badge-text">{leg}</text>
  </g>
</svg>"""
