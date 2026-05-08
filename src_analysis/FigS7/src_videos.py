import subprocess
from pathlib import Path

folders = [
    Path("/home/adrien/SPombe_MatRegion_Model/supp2polymers/2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullA_FullA_p2_0.00025_noise_500_swi6_400"),
    Path("/home/adrien/SPombe_MatRegion_Model/supp2polymers/2polymersFullAAFullMM_Triplicate1_simBis_1e7timesteps_FullM_FullM_p2_0.00025_noise_500_swi6_400")
]

for folder in folders:
    print(f"Running job in {folder}")

    result = subprocess.run(
        ["python", "src_ovito_video.py"],
        cwd=folder
    )

    if result.returncode != 0:
        print(f"Job failed in {folder}")
        break

    print(f"Finished {folder}")

print("Done")