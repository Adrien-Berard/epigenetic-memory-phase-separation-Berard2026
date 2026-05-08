import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Layout parameters ─────────────────────────────────────────────────────────
INSET_FRAC   = 0.35   # inset occupies this fraction of the main axes width/height
INSET_PAD    = 0.03   # padding from corner (axes fraction)
BORDER_LW    = 4      # border line width (points)
GREEN        = '#009E73'
ORANGE       = '#E69F00'

full_without = '/home/adrien/19_03_Sim/WihtoutEpe1/WithoutBE/full_view_wihtout_epe1.png'
zoom_without = '/home/adrien/19_03_Sim/WihtoutEpe1/WithoutBE/zoom_view_wihtout_epe1.png'
full_2nuc    = '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/2_nucleation_sites/full_view_2nuc.png'
zoom_2nuc    = '/home/adrien/19_03_Sim/WithEpe1/2or3Nuc/2_nucleation_sites/zoom_view_2nuc.png'


def load_snapshot(path: str | None) -> np.ndarray | None:
    """
    Load a snapshot image from a PNG, JPG, or PDF file.
    For PDFs the first page is rasterised to a numpy RGBA array.
    Returns None if path is None or the file does not exist.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f" snapshot not found: {path}")
        return None

    if p.suffix.lower() == ".pdf":
        try:
            import fitz                          # PyMuPDF  (pip install pymupdf)
            doc  = fitz.open(str(p))
            page = doc[0]
            # render at 150 dpi (mat scale=150/72 ≈ 2.08)
            mat  = fitz.Matrix(500 / 72, 500 / 72)
            pix  = page.get_pixmap(matrix=mat, alpha=False)
            arr  = np.frombuffer(pix.samples, dtype=np.uint8)
            return arr.reshape(pix.height, pix.width, 3)
        except ImportError:
            pass
        try:
            from pdf2image import convert_from_path   # pip install pdf2image
            imgs = convert_from_path(str(p), dpi=150, first_page=1, last_page=1)
            return np.array(imgs[0])
        except ImportError:
            print(f"  PDF snapshot needs PyMuPDF or pdf2image: pip install pymupdf")
            return None

    return plt.imread(str(p))

def crop_img(img, f=0.08):
    h, w = img.shape[:2]
    dy, dx = int(h * f), int(w * f)
    # crop bottom only
    return img[dy:h-dy, dx:w-dx]


img_without = crop_img(load_snapshot(full_without))
img_zoom_without = load_snapshot(zoom_without)
img_2nuc = crop_img(load_snapshot(full_2nuc))
img_zoom_2nuc = load_snapshot(zoom_2nuc)


def add_inset(fig, ax_main, zoom_img, color, corner='top-right'):
    """
    Add a zoom inset in a corner of ax_main with a reliable coloured border.
    corner: 'top-right' or 'bottom-right'
    """
    bbox = ax_main.get_position()   # figure-fraction Bbox

    w_inset = bbox.width  * INSET_FRAC
    h_inset = bbox.height * INSET_FRAC
    pad_x   = INSET_PAD  * bbox.width
    pad_y   = INSET_PAD  * bbox.height

    x0 = bbox.x1 - w_inset - pad_x

    if corner == 'top-right':
        y0 = bbox.y1 - h_inset - pad_y
    else:  # bottom-right
        y0 = bbox.y0 + pad_y

    ax_inset = fig.add_axes([x0, y0, w_inset, h_inset])
    ax_inset.imshow(zoom_img, interpolation='antialiased')
    ax_inset.set_axis_off()
    # ── Add labels ────────────────────────────────────────────────────────────────
    TOP_Y    = 1   # push above image (increase for more space)
    BOTTOM_Y = 0  # push below image (more negative = more space)

    axes[0].text(
        0.5, TOP_Y,
        rf"$epe1 \Delta$",style='italic',
        transform=axes[0].transAxes,
        ha='center', va='bottom',
        fontsize=18, font="serif"
    )

        # ── Reliable border: Rectangle drawn in figure-fraction coordinates ────────
    rect = plt.Rectangle(
        (x0, y0), w_inset, h_inset,
        linewidth=BORDER_LW,
        edgecolor=color,
        facecolor='none',
        transform=fig.transFigure,
        clip_on=False,
        zorder=10
    )
    fig.add_artist(rect)

    axes[1].annotate(
    "2cenH",
    xy=(x0+w_inset, h_inset),          # arrow target (center of image)
    xytext=(0.5, BOTTOM_Y),    # text position (same as before)
    xycoords='axes fraction',
    textcoords='axes fraction',
    ha='center', va='top',
    fontsize=18,font="serif",
    arrowprops=dict(
        arrowstyle='->',
        lw=2,
        color='black',
        shrinkA=5,
        shrinkB=5
    )
)

    return ax_inset


# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(10, 12),
                         gridspec_kw={'hspace': 0.04})

for ax in axes:
    ax.set_axis_off()

axes[0].imshow(crop_img(img_without), interpolation='antialiased')
axes[1].imshow(crop_img(img_2nuc),    interpolation='antialiased')

# Flush layout so ax.get_position() is accurate
fig.tight_layout(pad=0.5)
fig.canvas.draw()

add_inset(fig, axes[0], img_zoom_without, GREEN,  corner='top-right')
add_inset(fig, axes[1], img_zoom_2nuc,    ORANGE, corner='bottom-right')

# ── Save ──────────────────────────────────────────────────────────────────────
out = 'epe1_snapshots.pdf'
fig.savefig(out, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f"Saved → {out}")