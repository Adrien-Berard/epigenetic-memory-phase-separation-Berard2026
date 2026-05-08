import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import PythonScriptModifier
from ovito.vis import Viewport, OSPRayRenderer,ColorLegendOverlay, TextLabelOverlay
from ovito.data import Bonds
from ovito.pipeline import Pipeline
from ovito.qt_compat import QtCore

FILE_PATH = ('dump.lammpstrj')

# ── User settings ──────────────────────────────────────────────────
RENDER_FRAME  = 0
OUTPUT_IMAGE  = 'prxlife_render.png'
IMAGE_SIZE    = (1200, 900)
PARTICLE_RADIUS      = 0.5
BOND_RADIUS_FRACTION = 0.35

# Chromatin bead types — these are the beads that form the polymer
# backbone. Their type ID changes over time (A=1, U=2, M=3) but they
# are ALWAYS the first N_CHROMATIN particle IDs in the dump.
POLYMER_TYPES  = {1, 2, 3}   # all chromatin states
N_CHROMATIN    = 80          # fixed chain length — set this manually
# Bonds connect bead i → bead i+1 by sorted Particle Identifier,
# selecting only beads whose type is in POLYMER_TYPES at that frame.
# Since total chromatin count is fixed, CHAIN_LENGTHS = [N_CHROMATIN].
CHAIN_LENGTHS  = [N_CHROMATIN]

RADIUS_A = 0.5
PROTEIN_RADIUS = 0.5
POLYMER_RADII  = {1: RADIUS_A, 2: RADIUS_A, 3: RADIUS_A}   # chromatin
# ROTEIN_RADIUS = RADIUS_A/2                           # Swi6, Epe1, etc.



# Human-readable labels
TYPE_NAMES = {
    1: 'A',
    2: 'U',
    3: 'M',
    4: 'Swi6',
    5: 'Swi6*HDAC',
    6: 'Swi6*Epe1',
}
# ── PRX Life / Okabe-Ito palette ───────────────────────────────────
def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4))

PRX_COLORS = {
    1: _hex_to_rgb('#0072B2'),   # blue       
    2: _hex_to_rgb("#E7E078"),   # yellow     
    3: _hex_to_rgb("#FF0000"),   # red       
    4: _hex_to_rgb('#CC79A7'),   # pink       
    5: _hex_to_rgb('#009E73'),   # green      
    6: _hex_to_rgb('#E69F00'),   # orange     
#     7: _hex_to_rgb('#56B4E9'),
#     8: _hex_to_rgb('#000000'),
#     9: _hex_to_rgb('#999999'),
#    10: _hex_to_rgb('#4D4D4D'),
}

BOND_COLOR      = _hex_to_rgb("#D3D3D3") # Light-grey
CELL_LINE_COLOR = _hex_to_rgb('#AAAAAA')
# ── Inspect: scan frames to find ALL types that ever appear ───────
# Types evolve during the sim (beads switch A/U/M, Swi6 binds/unbinds),
# so frame 0 alone won't show the full type set.
pipeline_inspect = import_file(
    FILE_PATH,
    columns=['Particle Identifier', 'Particle Type',
             'Position.X', 'Position.Y', 'Position.Z']
)
n_frames = pipeline_inspect.source.num_frames
print(f'Total frames in trajectory: {n_frames}')

# Sample up to 20 frames evenly spaced to discover all type IDs
sample_frames = np.unique(np.linspace(0, n_frames - 1, min(20, n_frames), dtype=int))
all_type_ids = set()
for f in sample_frames:
    d = pipeline_inspect.compute(f)
    ids = {t.id for t in d.particles['Particle Type'].types}
    all_type_ids |= ids

print(f'Type IDs found across sampled frames: {sorted(all_type_ids)}')
print()

# Show counts at first and last sampled frame for reference
for check_frame in [sample_frames[0], sample_frames[-1]]:
    d = pipeline_inspect.compute(check_frame)
    ptypes = d.particles['Particle Type']
    print(f'Frame {check_frame}:')
    for t in ptypes.types:
        count = int(np.sum(np.array(ptypes) == t.id))
        label = TYPE_NAMES.get(t.id, '?')
        print(f'  type {t.id} ({label}): {count} particles')
    print()

d0 = pipeline_inspect.compute(0)
box = d0.cell.matrix
print(f'Box: {box[0,0]:.1f} x {box[1,1]:.1f} x {box[2,2]:.1f}')
print(f'Total particles per frame: {d0.particles.count} (fixed)')
pipeline = import_file(
    FILE_PATH,
    columns=['Particle Identifier', 'Particle Type',
             'Position.X', 'Position.Y', 'Position.Z']
)

# ── Modifier 1: assign PRX colors — robust to evolving types ──────
# Problem: if a type first appears at frame 10, it won't be in the
# type list at frame 0. OVITO adds new ParticleType entries lazily
# as it encounters them, so we must handle types being absent.
#
# Strategy: for every type in PRX_COLORS that is NOT yet in the
# property's type list, we add it explicitly so colors are always
# pre-registered. Types present but not in PRX_COLORS get grey.


def assign_colors(frame, data):
    pt_prop = data.particles_['Particle Type_']  # mutable

    # IDs already known at this frame
    existing_ids = {t.id for t in pt_prop.types}

    # Pre-register any type from our palette not yet seen
    for type_id, color in PRX_COLORS.items():
        if type_id not in existing_ids:
            pt_prop.add_type_id(type_id,data.particles_)  # register with default props

    # Now assign colors, radii, and names to all known types
    for t in pt_prop.types_:
        t.name  = TYPE_NAMES.get(t.id, str(t.id))
        t.color  = PRX_COLORS.get(t.id, (1, 0.0, 0.0))
        t.radius = POLYMER_RADII.get(t.id, PROTEIN_RADIUS)
        

pipeline.modifiers.append(PythonScriptModifier(function=assign_colors))

# Modifier 2

def create_backbone_bonds(frame, data):
    pids   = data.particles['Particle Identifier']
    ptypes = data.particles['Particle Type']
    pos    = np.array(data.particles['Position'])  # (N, 3) absolute coords
    cell   = data.cell.matrix[:3, :3]              # 3x3 box vectors

    id_to_idx = {int(pid): i for i, pid in enumerate(pids)}

    polymer_mask = np.isin(ptypes, sorted(POLYMER_TYPES))
    polymer_ids  = np.sort(pids[polymer_mask])
    N_poly       = len(polymer_ids)

    if N_poly == 0:
        print(f'[WARN] frame {frame}: no polymer beads found')
        return

    chain_lengths = list(CHAIN_LENGTHS)
    if sum(chain_lengths) != N_poly:
        print(f'[WARN] frame {frame}: expected {sum(chain_lengths)}, found {N_poly}')
        chain_lengths = [N_poly]

    cell_inv = np.linalg.inv(cell)
    bond_pairs, pbc_images, cursor = [], [], 0

    for clen in chain_lengths:
        chain = polymer_ids[cursor:cursor + clen]
        for j in range(len(chain) - 1):
            a = id_to_idx[int(chain[j])]
            b = id_to_idx[int(chain[j + 1])]
            bond_pairs.append([a, b])

            # Fractional displacement → nearest image
            # image[k] tells OVITO: draw bond to b + image[k]*cell[k]
            dr_frac = cell_inv @ (pos[b] - pos[a])
            image   = -np.round(dr_frac).astype(np.int32)
            pbc_images.append(image)

        cursor += clen

    bond_arr  = np.array(bond_pairs,  dtype=np.int32)
    image_arr = np.array(pbc_images,  dtype=np.int32)  # shape (N_bonds, 3)

    data.particles_.create_bonds(count=len(bond_arr))
    data.particles_.bonds_.create_property('Topology',       data=bond_arr)
    data.particles_.bonds_.create_property('Periodic Image', data=image_arr)
    data.particles_.bonds_.create_property(
        'Color', data=np.tile(BOND_COLOR, (len(bond_arr), 1)))

    bv = data.particles_.bonds_.vis
    bv.width   = RADIUS_A * BOND_RADIUS_FRACTION  # matches chromatin radius
    bv.shading = bv.Shading.Normal
pipeline.modifiers.append(PythonScriptModifier(function=create_backbone_bonds))

# ── Modifier 3: style the simulation cell ─────────────────────────
def style_cell(frame, data):
    if data.cell is None:
        return
    cv = data.cell_.vis
    cv.enabled         = True
    cv.line_width      = PARTICLE_RADIUS * 0.1
    cv.rendering_color = CELL_LINE_COLOR

pipeline.modifiers.append(PythonScriptModifier(function=style_cell))

# ── Verify pipeline runs cleanly before rendering ─────────────────

print(f'pipeline modifiers: {len(pipeline.modifiers)}')
data = pipeline.compute(RENDER_FRAME)
n_bonds = data.particles.bonds.count if data.particles.bonds else 0
print(f'Particles: {data.particles.count}, Bonds: {n_bonds}')
for t in data.particles['Particle Type'].types:
    print(f'  type {t.id} ({t.name}): color={tuple(round(float(c),3) for c in t.color)}')



def modify(frame, data):
    data.attributes["Time_k"] = float(data.attributes["Timestep"]) / 1000.0


# ── Render ────────────────────────────────────────────────────────
# Clear all modifiers and re-add them cleanly
pipeline.modifiers.clear()
pipeline.modifiers.append(PythonScriptModifier(function=assign_colors))      # your fixed version
pipeline.modifiers.append(PythonScriptModifier(function=create_backbone_bonds))
pipeline.modifiers.append(PythonScriptModifier(function=style_cell))
pipeline.modifiers.append(PythonScriptModifier(function=modify))

pipeline.add_to_scene()
PythonScriptModifier(function=create_backbone_bonds).enabled = True

# vp = Viewport()
# vp.type       = Viewport.Type.Perspective
# # vp.camera_pos = (25, -100, 25) # Full view
# vp.camera_pos = (35, 16, 35) # Zoom
# vp.camera_dir = (0, 1, 0)
# vp.camera_up  = (0, 0, 1)

# # ── OSPRay: publication still image ────────────────────────────────
# renderer_image = OSPRayRenderer()
# renderer_image.samples_per_pixel         = 8      # default, good balance
# renderer_image.refinement_iterations     = 8      # more passes = cleaner
# renderer_image.denoising_enabled         = True   # AI denoise, nearly free
# renderer_image.ambient_light_enabled     = True
# renderer_image.ambient_brightness        = 1
# renderer_image.direct_light_enabled      = True
# renderer_image.direct_light_intensity    = 1
# renderer_image.direct_light_angular_diameter = np.radians(45.0)  # soft shadows
# renderer_image.material_type             = 'standard'
# renderer_image.principled_roughness      = 0.2   # slight sheen on spheres
# renderer_image.principled_metalness      = 0.2  # mostly diffuse, tiny metallic
# renderer_image.principled_specular_brightness = 0.6

# vp.render_image(filename='M_state_polymer_zoom.png', size=(2400, 2400),
#                 frame=10, renderer=renderer_image, alpha=False)

# vp = Viewport()
# vp.type       = Viewport.Type.Perspective
# vp.camera_pos = (25, -100, 25) # Full view
# vp.camera_pos = (30, -5, 30) # Zoom
# vp.camera_dir = (0, 1, 0)
# vp.camera_up  = (0, 0, 1)

# # ── OSPRay: publication still image ────────────────────────────────
# renderer_image = OSPRayRenderer()
# renderer_image.samples_per_pixel         = 8      # default, good balance
# renderer_image.refinement_iterations     = 8      # more passes = cleaner
# renderer_image.denoising_enabled         = True   # AI denoise, nearly free
# renderer_image.ambient_light_enabled     = True
# renderer_image.ambient_brightness        = 1
# renderer_image.direct_light_enabled      = True
# renderer_image.direct_light_intensity    = 1
# renderer_image.direct_light_angular_diameter = np.radians(45.0)  # soft shadows
# renderer_image.material_type             = 'standard'
# renderer_image.principled_roughness      = 0.2   # slight sheen on spheres
# renderer_image.principled_metalness      = 0.2  # mostly diffuse, tiny metallic
# renderer_image.principled_specular_brightness = 0.6

# vp.render_image(filename='A_state_polymer_zoom.png', size=(2400, 2400),
#                 frame=300000, renderer=renderer_image, alpha=False)

# # ── OSPRay: video frames ────────────────────────────────────────────
overlay = TextLabelOverlay(
    text="Time: [Time_k] " + "(τ_LJ)", text_color = (0,0,0), font_size = 0.03)
# Specify the source of dynamically computed attributes.
overlay.pipeline = pipeline

vp = Viewport()
vp.type       = Viewport.Type.Perspective
vp.camera_pos = (25, -100, 25) # Full view
# vp.camera_pos = (30, -5, 30) # Zoom
vp.camera_dir = (0, 1, 0)
vp.camera_up  = (0, 0, 1)

vp.overlays.append(overlay)

# # ── OSPRay: publication still image ────────────────────────────────
renderer_video = OSPRayRenderer()
renderer_video.samples_per_pixel         = 4      # default, good balance
renderer_video.refinement_iterations     = 4      # more passes = cleaner
renderer_video.denoising_enabled         = True   # AI denoise, nearly free
renderer_video.ambient_light_enabled     = True
renderer_video.ambient_brightness        = 1
renderer_video.direct_light_enabled      = True
renderer_video.direct_light_intensity    = 1
renderer_video.direct_light_angular_diameter = np.radians(45.0)  # soft shadows
renderer_video.material_type             = 'standard'
renderer_video.principled_roughness      = 0.2   # slight sheen on spheres
renderer_video.principled_metalness      = 0.2  # mostly diffuse, tiny metallic
renderer_video.principled_specular_brightness = 0.6

legend = ColorLegendOverlay(
    title = ' ',
    alignment = QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignTop,
    orientation = QtCore.Qt.Orientation.Vertical,
    offset_y = -0.04,
    font_size = 0.08,
    label_size = 0.6)

legend.property = 'particles/Particle Type'

vp.overlays.append(legend)

vp.render_anim(filename='bistability_movie_2transitions.mp4', size=(1400, 1400), fps=8, range =[4000,10000],
               every_nth=10, renderer=renderer_video,
                ffmpeg_executable="ffmpeg",  
                ffmpeg_codec="libx264",
                ffmpeg_quality="low")

vp.render_anim(filename='full_bistability.mp4', size=(1400, 1400), fps=8,
                every_nth=40, range = [10,n_frames],renderer=renderer_video,
                ffmpeg_executable="ffmpeg", 
                ffmpeg_codec="libx264",
                ffmpeg_quality="low") 

