"""
Example script for setting policy and running OG-Core.
"""

# import modules
import multiprocessing
from distributed import Client
import time
import numpy as np
import os
from ogcore import output_tables as ot
from ogcore import output_plots as op
from ogcore.execute import runner
from ogcore.parameters import Specifications
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore.utils import safe_read_pickle
import ogcore
from ogcore import SS
import matplotlib.pyplot as plt
import pandas as pd
from tools.utils import get_param
#from tools.params import update_params, interpolate_partial,resample_omega
from tools.nz import nz_params
from copy import deepcopy

# Use a custom matplotlib style file for plots
plt.style.use("ogcore.OGcorePlots")


def main():
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(CUR_DIR, "results")
    base_dir = os.path.join(save_dir, BASELINE_DIR)

    # Directories to save data
    p = obtain_base_params(base_dir, num_workers = 1)
    runner(p, time_path=True, client=None)

    p2 = obtain_reform_params(p, save_dir)
    runner(p2, time_path=True, client=None)

    postp(save_dir, p.start_year)


def postp(save_dir, start_year):
    base_dir = os.path.join(save_dir, BASELINE_DIR)
    reform_dir = os.path.join(save_dir, REFORM_DIR)
    # return ans - the percentage changes in macro aggregates and prices
    # due to policy changes from the baseline to the reform
    base_tpi = safe_read_pickle(os.path.join(base_dir, "TPI", "TPI_vars.pkl"))
    base_params = safe_read_pickle(os.path.join(base_dir, "model_params.pkl"))
    reform_tpi = safe_read_pickle(
        os.path.join(reform_dir, "TPI", "TPI_vars.pkl")
    )
    reform_params = safe_read_pickle(
        os.path.join(reform_dir, "model_params.pkl")
    )
    ans = ot.macro_table(
        base_tpi,
        base_params, 
        reform_tpi=reform_tpi,
        reform_params=reform_params,
        var_list=["Y", "C", "K", "L", "r", "w"],
        output_type="pct_diff",
        num_years=10,
        start_year=start_year,
    )

    # create plots of output
    op.plot_all(
        base_dir, reform_dir, os.path.join(save_dir, "OG-Core_example_plots")
    )
    ans.to_csv(os.path.join(save_dir, "OG-Core_example_output.csv"))
    
    print("Job done !")


def obtain_base_params(save_dir, num_workers = 1):

    base_dir = os.path.join(save_dir, BASELINE_DIR)

    # Set some OG model parameters
    # See default_parameters.json for more description of these parameters
    # ------------------------------------------------------------------------
    # Run baseline policy first
    # ------------------------------------------------------------------------
    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=base_dir,
        output_base=base_dir,
    )

    p = nz_params(p, save_dir)

    return p

def obtain_reform_params(p_base, save_dir):

    p_reform = deepcopy(p_base)

    reform_dir = os.path.join(save_dir, REFORM_DIR)
    p_reform.output_base = reform_dir
    p_reform.baseline = False
    # p_reform.alpha_I = 0.5 * p_base.alpha_I
    p_reform.debt_ratio_ss = 1.5 * p_base.debt_ratio_ss

    return p_reform

if __name__ == "__main__":
    # execute only if run as a script
    main()
