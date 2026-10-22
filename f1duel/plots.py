"""One figure style for the whole project (colour-blind-safe palette, validated)."""
import matplotlib as mpl
import matplotlib.pyplot as plt

INK, INK2, MUTED, GRID = "#1f1f1e", "#55554f", "#8a8a84", "#e6e5e0"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b1"
TIER_COLORS = ["#104281", "#2a78d6", "#9ec5f4"]          # top / midfield / back (one hue, dark -> light)


def style():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def title(ax, text):
    ax.set_title(text, loc="left", color=INK)


def save(fig, name):
    from .data import ROOT
    (ROOT / "figures").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figures" / f"{name}.png")
