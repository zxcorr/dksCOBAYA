"""
Usage:
    python run_dksCOBAYA.py                      # Runs Pantheon+ (default)
    python run_dksCOBAYA.py run_pantheon.yaml     # Runs Pantheon+
    python run_dksCOBAYA.py run_desi_dr2.yaml     # Runs DESI DR2 BAO
    python run_dksCOBAYA.py run_joint.yaml        # Runs SNe + BAO jointly

P.s.[AM]: I have included our dksMODEL in the theory folder and then we do not need to force the location anymore
"""
import sys
from cobaya.run import run
from cobaya.yaml import yaml_load_file

if __name__ == "__main__":
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "run_pantheon.yaml"

    print(f"=== Loading configuration: {yaml_file} ===")
    info = yaml_load_file(yaml_file)
    print("=== Starting MCMC ===")
    updated_info, sampler = run(info)
    print("=== Done ===")