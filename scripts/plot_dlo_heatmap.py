"""
Generate a heatmap of mean ΔLogOdds by pipeline stage × challenge type × context.
Data taken directly from Table 1 of the paper (tab:dlo_type_ctx_comp).
Produces: paper/figures/dlo_type_context_heatmap.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Data from Table 1 (tab:dlo_type_ctx_comp)
# Column order: [Simp-IC, Simp-PE, Ethos-IC, Ethos-PE, Just-IC, Just-PE, Cite-IC, Cite-PE]
# ---------------------------------------------------------------------------
ROW_LABELS = [
    "OLMo 3 Base",
    "Think SFT",
    "Think DPO",
    "Think",
    "Instruct SFT",
    "Instruct DPO",
    "Instruct",
    "Llama 3.1 Base",
    "Llama 3.1 Instruct",
    "Tulu 3 SFT",
    "Tulu 3 DPO",
    "Tulu 3",
]

COMP = np.array([
    [-0.229,  0.002,  0.111,  0.373,  0.134,  0.401,  0.159,  0.429],
    [-0.250,  0.026,  0.209,  0.583,  0.209,  0.523,  0.279,  0.544],
    [-0.236,  0.027,  0.196,  0.584,  0.201,  0.520,  0.278,  0.544],
    [-0.241,  0.027,  0.196,  0.596,  0.204,  0.532,  0.289,  0.556],
    [-0.158,  0.040,  0.186,  0.676,  0.253,  0.608,  0.322,  0.588],
    [-0.085,  0.055,  0.126,  0.781,  0.215,  0.654,  0.345,  0.684],
    [-0.095,  0.063,  0.101,  0.838,  0.214,  0.691,  0.369,  0.716],
    [-0.276,  0.000,  0.098,  0.394,  0.118,  0.401,  0.152,  0.458],
    [-0.261, -0.013,  0.149,  0.451,  0.159,  0.450,  0.169,  0.478],
    [-0.336, -0.012,  0.131,  0.468,  0.108,  0.440,  0.110,  0.462],
    [-0.519, -0.020,  0.070,  0.810, -0.020,  0.698, -0.013,  0.759],
    [-0.545, -0.029,  0.020,  0.844, -0.076,  0.706, -0.065,  0.781],
])

MED = np.array([
    [-0.169,  0.015,  0.319,  0.789,  0.325,  0.778,  0.333,  0.794],
    [-0.206,  0.014,  0.324,  0.858,  0.334,  0.857,  0.361,  0.879],
    [-0.220,  0.015,  0.325,  0.885,  0.338,  0.893,  0.368,  0.908],
    [-0.219,  0.017,  0.330,  0.892,  0.342,  0.899,  0.376,  0.917],
    [-0.248,  0.011,  0.296,  0.855,  0.307,  0.852,  0.337,  0.872],
    [-0.300,  0.022,  0.226,  0.917,  0.364,  1.038,  0.405,  1.107],
    [-0.305,  0.030,  0.241,  0.954,  0.397,  1.107,  0.418,  1.155],
    [-0.084,  0.011,  0.448,  0.863,  0.446,  0.871,  0.458,  0.880],
    [-0.069,  0.012,  0.381,  0.820,  0.387,  0.823,  0.397,  0.845],
    [-0.065,  0.008,  0.425,  0.859,  0.426,  0.866,  0.435,  0.877],
    [-0.085,  0.002,  0.464,  1.039,  0.469,  1.093,  0.487,  1.103],
    [-0.095,  0.001,  0.456,  1.040,  0.458,  1.091,  0.480,  1.102],
])

# Column groupings: (label, [IC_col_idx, PE_col_idx])
TYPE_GROUPS = [
    ("Simple",        [0, 1]),
    ("Ethos",         [2, 3]),
    ("Justification", [4, 5]),
    ("Citation",      [6, 7]),
]
COL_LABELS = ["IC", "PE", "IC", "PE", "IC", "PE", "IC", "PE"]

# Dividers between pipeline groups (after row index)
PIPELINE_DIVIDERS = [0, 3, 6, 7, 9]  # after OLMo Base, after Think, after Instruct, after Llama Base

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
VMIN, VMAX = -0.6, 1.2
norm = TwoSlopeNorm(vmin=VMIN, vcenter=0, vmax=VMAX)
cmap = "RdBu_r"

fig, axes = plt.subplots(
    1, 2,
    figsize=(14, 5.2),
    gridspec_kw={"wspace": 0.06},
)

def draw_panel(ax, data, title):
    n_rows, n_cols = data.shape
    im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")

    # Cell annotations
    for r in range(n_rows):
        for c in range(n_cols):
            val = data[r, c]
            # White text on dark cells, dark on light
            brightness = norm(val)
            text_color = "white" if abs(brightness - 0.5) > 0.28 else "black"
            ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                    fontsize=6.5, color=text_color, fontweight="normal")

    # Axes ticks
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(COL_LABELS, fontsize=8)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(ROW_LABELS, fontsize=8.5)
    ax.tick_params(axis="both", length=0)

    # Type-group labels below IC/PE tick labels
    for label, (ci, cj) in [(g[0], (g[1][0], g[1][-1])) for g in TYPE_GROUPS]:
        mid = (ci + cj) / 2.0
        ax.text(mid, -0.12, label, ha="center", va="top",
                fontsize=8, fontweight="bold",
                transform=ax.get_xaxis_transform())

    # Horizontal dividers between pipeline groups
    for after_row in PIPELINE_DIVIDERS:
        ax.axhline(y=after_row + 0.5, color="0.35", lw=1.0, ls="-")

    ax.set_title(title, fontsize=11, fontweight="bold", pad=26)
    return im

im = draw_panel(axes[0], COMP, "Computational")
draw_panel(axes[1], MED, "Medical")

# Hide y-tick labels on right panel
axes[1].set_yticklabels([])

# Colorbar
cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation="vertical",
    fraction=0.018,
    pad=0.02,
    shrink=0.92,
)
cbar.set_label("Mean $\\Delta$LogOdds", fontsize=9)
cbar.ax.tick_params(labelsize=8)

out_path = os.path.join(
    os.path.dirname(__file__), "..", "paper", "figures", "dlo_type_context_heatmap.png"
)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Saved: {os.path.abspath(out_path)}")
