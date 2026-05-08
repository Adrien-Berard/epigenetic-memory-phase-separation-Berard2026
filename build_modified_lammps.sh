#!/usr/bin/env bash
set -e  # stop on error
set -x  # print every command before executing it

# --- Clone LAMMPS at the specific tag ---
trash modified_lammps_Apr2024

git clone --depth 1 --branch patch_17Apr2024 \
    https://github.com/lammps/lammps.git modified_lammps_Apr2024

cd modified_lammps_Apr2024
git switch -c my_modified_lammps

# --- Clean previous builds ---
trash build

# --- Replace REACTION fix ---
trash src/REACTION/fix_bond_react.cpp
trash src/REACTION/fix_bond_react.h

base_react="https://raw.githubusercontent.com/Adrien-Berard/epigenetic-memory-phase-separation-Berard2026/master/fix_bond_react_modified_version_Apr2024"
wget -q "${base_react}/fix_bond_react.cpp" -O src/REACTION/fix_bond_react.cpp
wget -q "${base_react}/fix_bond_react.h"   -O src/REACTION/fix_bond_react.h


# --- Build ---
mkdir build
cd build

cmake -C ../cmake/presets/most.cmake \
      -C ../cmake/presets/nolib.cmake \
      -D BUILD_MPI=ON \
      -D PKG_REACTION=ON \
      -D PKG_MOLECULE=ON \
      ../cmake

cmake --build . -j$(nproc)

# --- Verify ---
./lmp -h

# --- Add lmp to PATH ---
echo "export PATH=\$PATH:/home/../modified_lammps_Apr2024/build" >> ~/.bashrc  # Change your path accordingly
source ~/.bashrc
lmp -h
