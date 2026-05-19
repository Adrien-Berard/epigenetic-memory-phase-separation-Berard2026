# Epigenetic memory achieved through chromatin-induced phase separation

Analysis code and interactive mean-field demo accompanying **Berard et al. (2026)**.

Simulation trajectories and scan outputs are **not stored in this repository**. Download the datasets from **Zenodo** (https://doi.org/10.5281/zenodo.16911423) and set `BERARD_DATA_ROOT` to the unpacked folder, or place files under `./data/` following the layout described on Zenodo.

---

## Interactive mean-field model

Open the two-state nucleosome Gillespie demo in a browser (sliders for rates and initial conditions):

**[Mean-field demo — MeanField_Epigenetic_Two-state.html](MeanField_Epigenetic_Two-state.html)**

---

## Repository layout

| Path | Role |
|------|------|
| [`src_analysis/`](src_analysis/) | Figure-building scripts (one folder per main or supplementary figure) |
| [`src_analysis/collect_and_zip.py`](src_analysis/collect_and_zip.py) | Zip LAMMPS run folders from a scan tree (`--root`, `--out`) |
| [`MeanField_Epigenetic_Two-state.html`](MeanField_Epigenetic_Two-state.html) | Browser mean-field / Gillespie explorer |
| [`fix_bond_react_modified_version_Apr2024/`](fix_bond_react_modified_version_Apr2024/) | Modified LAMMPS `fix bond/react` by Jacob Gissinger (April 2024 base) |
| [`build_modified_lammps.sh`](build_modified_lammps.sh) | Build script for the patched LAMMPS |

---

## `src_analysis` — scripts by figure

Main figure compositors are named `src_fig*.py`. OVITO render scripts (`src_ovito*.py`) are run inside simulation directories. Figure 8 uses [`mean-field.py`](src_analysis/Fig8/mean-field.py) (static panels); the HTML file is the interactive companion.

| Folder | Entry script(s) | Purpose |
|--------|-----------------|--------|
| **Fig2** | [`src_fig2.py`](src_analysis/Fig2/src_fig2.py) | Figure 2: Rg/types, model PDFs, snapshots, displacement histogram |
| | [`src_ovito.py`](src_analysis/Fig2/src_ovito.py) | OVITO renders |
| **Fig3** | [`src_fig3.py`](src_analysis/Fig3/src_fig3.py) | Figure 3: model panel, snapshots, time series (kymograph, polymer counts, Swi6, Rg; no timeline) |
| | `src_ovito.py`, `src_ovito_zoom.py` | OVITO renders |
| **Fig4** | [`src_fig4.py`](src_analysis/Fig4/src_fig4.py) | Figure 4: phase-scan composite (`all_results.csv` in folder) |
| **Fig5** | [`src_fig5.py`](src_analysis/Fig5/src_fig5.py) | Figure 5: cell-cycle composite |
| | [`CellCycle/src_ovito_video.py`](src_analysis/Fig5/CellCycle/src_ovito_video.py) | OVITO movies |
| **Fig6** | [`src_fig6.py`](src_analysis/Fig6/src_fig6.py) | Figure 6: two-polymer fixed vs diffusive |
| | [`src_ovito_video.py`](src_analysis/Fig6/src_ovito_video.py) | OVITO movies |
| **Fig7** | [`src_fig7.py`](src_analysis/Fig7/src_fig7.py) | Figure 7: model panel, kymographs, Hi-C, Epe1 nucleation snapshots |
| | [`epe1_snapshots.py`](src_analysis/Fig7/epe1_snapshots.py) | Snapshot layout helper |
| | [`src_ovito_3nuc.py`](src_analysis/Fig7/src_ovito_3nuc.py) | OVITO renders |
| **Fig8** | [`mean-field.py`](src_analysis/Fig8/mean-field.py) | Gillespie panels (see HTML for interactive version) |
| **FigS1** | [`potentials.ipynb`](src_analysis/FigS1/potentials.ipynb) | Morse / smooth-linear potentials |
| **FigS2** | [`src_figS2.py`](src_analysis/FigS2/src_figS2.py) | Supplementary polymer time series |
| **FigS3** | [`src_figS3.py`](src_analysis/FigS3/src_figS3.py) | Small time-series panels and RGB scan results (`all_results.csv`) |
| **FigS4** | [`src_figS4_k1.py`](src_analysis/FigS4/src_figS4_k1.py), [`src_figS4_k2.py`](src_analysis/FigS4/src_figS4_k2.py) | Phase-diagram scans: k1 vs Swi6 and k2 vs Swi6 |
| **FigS5** | [`src_figS5.py`](src_analysis/FigS5/src_figS5.py), [`src_figS5_fine.py`](src_analysis/FigS5/src_figS5_fine.py) | Phase-diagram scans: k1 vs k2 (coarse and fine grids) |
| **FigS6** | [`src_figS6.py`](src_analysis/FigS6/src_figS6.py) | Placeholder for supplementary figure (code to be added)|
| **FigS7** | [`src_figS7.py`](src_analysis/FigS7/src_figS7.py) | Timescale / reaction-rate supplementary figure |
| **FigS8** | [`src_figS8.py`](src_analysis/FigS8/src_figS8.py) | Switching-polymers time series |
| | [`src_videos.py`](src_analysis/FigS8/src_videos.py) | Batch OVITO video driver |
| **FigS9** | [`src_figS9.py`](src_analysis/FigS9/src_figS9.py) | Diffusive two-polymer supplementary composite |
| | `src_ovito_FullA.py`, `src_ovito_FullM.py` | OVITO renders |
| **FigS10** | [`src_figS10.py`](src_analysis/FigS10/src_figS10.py), [`src_figS10_contact.py`](src_analysis/FigS10/src_figS10_contact.py) | Replicated time series + contact maps / HTML slider |
| **FigS11** | [`src_figS11.py`](src_analysis/FigS11/src_figS11.py), [`src_figS11_contact.py`](src_analysis/FigS11/src_figS11_contact.py) | Replicated time series + contact maps / HTML slider |
| **FigS12** | [`src_figS12.py`](src_analysis/FigS12/src_figS12.py), [`src_figS12_contact.py`](src_analysis/FigS12/src_figS12_contact.py) | Replicated time series + contact maps / HTML slider |
| **FigS13** | [`src_figS13.py`](src_analysis/FigS13/src_figS13.py), [`src_figS13_replication.py`](src_analysis/FigS13/src_figS13_replication.py) | Replicated time series + replication-phase contact maps |
| | `src_ovito_video.py` (per folder) | OVITO movies |

---

## Data paths

Scripts that read simulation outputs use:

```python
DATA_ROOT = Path(os.environ.get("BERARD_DATA_ROOT", "./data"))
```

Point `BERARD_DATA_ROOT` at your Zenodo unpack (e.g. `export BERARD_DATA_ROOT=/path/to/zenodo`). Each archive below is a separate download; after unpacking, the layout is:

```
zenodo/
├── Fig2.zip  →  LeftPanel/
│                RightPanel/
├── Fig3.zip
├── Fig4.zip/
│   ├── Noise500/
│   │   ├── FullA/          … simulation run folders
│   │   └── FullM/          … simulation run folders
│   └── ParameterScanDifferentSwi6/
│       ├── sim_p2_0.00025_noise_500_swi6_200/
│       ├── sim_p2_0.00025_noise_500_swi6_400/
│       └── sim_p2_0.00025_noise_500_swi6_600/
├── Fig5.zip  →  CellCycle/
├── Fig6.zip/
│   ├── sim_p2_0.00025_noise_500_swi6_400_nuc_160/
│   └── sim_p2_0.00025_noise_500_swi6_400_nuc_160Fixed/
├── Fig7.zip/
│   ├── 2_nucleation_sites/
│   ├── 3_nucleation_sites/
│   ├── cenH/
│   └── withoutEpe1/
├── FigS2.zip
├── FigS3-4.zip/
│   ├── k2_swi6/
│   │   ├── Noise250/
│   │   │   ├── FullA/
│   │   │   └── FullM/
│   │   ├── Noise500/
│   │   │   ├── FullA/
│   │   │   └── FullM/
│   │   └── Noise1000/
│   │       ├── FullA/
│   │       └── FullM/
│   └── k1_swi6/
│       ├── Noise250/
│       │   ├── FullA/
│       │   └── FullM/
│       ├── Noise500/
│       │   ├── FullA/
│       │   └── FullM/
│       └── Noise1000/
│           ├── FullA/
│           └── FullM/
├── FigS5.zip/
│   ├── k1_k2/
│   │   ├── Noise250/
│   │   │   ├── FullA/
│   │   │   └── FullM/
│   │   ├── Noise500/
│   │   │   ├── FullA/
│   │   │   └── FullM/
│   │   └── Noise1000/
│   │       ├── FullA/
│   │       └── FullM/
│   └── k1_k2_fine/
│       └── Noise500/
│           ├── FullA/
│           └── FullM/
├── FigS6.zip/
│   ├── Same_cycle_length/
│   ├── Same_cycle_length_M/
│   └── Same_cycle_noise500/
├── FigS8.zip/
│   ├── 2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullA_FullA_p2_0.00025_noise_500_swi6_400/
│   └── 2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullM_FullM_p2_0.00025_noise_500_swi6_400/
├── FigS9.zip/
│   ├── equil_swi6_1000/
│   ├── equil_swi6_200/
│   ├── equil_swi6_600/
│   ├── sim_p2_0.00025_noise_500_swi6_1000_nuc_80/
│   ├── sim_p2_0.00025_noise_500_swi6_200_nuc_80/
│   └── sim_p2_0.00025_noise_500_swi6_600_nuc_80/
├──FigS10-13.zip/
│   ├── 10CyclcesWithout/
│   ├── 2nuc/
│   ├── 3nuc/
│   └── cenH/
└── SuppMovies.zip/
      ├── VideoS01.mp4
      ├── VideoS02.mp4
      ├── VideoS03.mp4
      ├── VideoS04.mp4
      ├── VideoS05.mp4
      ├── VideoS06.mp4
      ├── VideoS07.mp4
      ├── VideoS08.mp4
      ├── VideoS09.mp4
      ├── VideoS10.mp4
      ├── VideoS11.mp4
      ├── VideoS12.mp4
      └── VideoS13.mp4

```

Each `FullA/` and `FullM/` leaf contains one LAMMPS run per parameter point (input files, dumps, `types1.dat`, etc.). Figures **2** and **3** expect local filenames in the working directory (`left_r2.dat`, `id_and_type.dat`, …) unless paths are edited in the scripts.

---

## Dependencies

Typical stack: **Python 3.10+**, `numpy`, `matplotlib`, `pandas`, `scipy`, `tqdm`, `pdf2image` (where PDF snapshots are used). Phase scans use the standard library only. **OVITO** Python module for render scripts. Optional: `poppler` for `pdf2image`.

---

## Packaging simulation folders

```bash
python src_analysis/collect_and_zip.py --root /path/to/scan_folder --out Fig6.zip
```

---

## Citation

Berard et al., *Epigenetic memory achieved through chromatin-induced phase separation* (2026).  
*(Full bibliographic entry and Zenodo DOI(https://doi.org/10.5281/zenodo.16911423))*
