
import os
from ogcore.utils import safe_read_pickle
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore import output_tables as ot
from ogcore import output_plots as op

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