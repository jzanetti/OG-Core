"""
Example script for setting policy and running OG-Core.
"""

# import modules
import os
from ogcore.execute import runner

from ogcore.constants import BASELINE_DIR

import matplotlib.pyplot as plt

from dill import load as dill_load
from tools.params import obtain_reform_params, run_eval_wrapper
from tools.postp import postp
from shutil import rmtree
plt.style.use("ogcore.OGcorePlots")


def main(run_eval: bool = False, run_base: bool = False, run_reform: bool = False, new_run = False):
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "results")
    
    if new_run:
        rmtree(save_dir)

    if run_eval:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        run_eval_wrapper(save_dir)
    
    if run_base:
        p = dill_load(open(f"{save_dir}/optimal_params.pickle", "rb"))
        baseline_dir = os.path.join(save_dir, BASELINE_DIR)

        for dir_type in ["SS", "TPI"]:
            proc_dir = os.path.join(baseline_dir, dir_type)
            if not os.path.exists(proc_dir):
                os.makedirs(proc_dir)

        p.baseline_dir = baseline_dir
        p.output_base = baseline_dir
        runner(p, time_path=True, client=None)

    if run_reform:
        p2 = obtain_reform_params(p, save_dir)
        runner(p2, time_path=True, client=None)
        postp(save_dir, p.start_year)

if __name__ == "__main__":
    # execute only if run as a script
    main(run_eval = True, run_base = True, new_run = True)
