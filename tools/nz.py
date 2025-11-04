from tools.utils import get_param
from tools.params import update_params, interpolate_partial,resample_method1, resample_s
from numpy import asarray
from numpy import ones
def nz_params(p, save_dir):
    og_spec = update_params(p)

    # Update parameters for baseline from default json file
    # p.update_specifications(og_spec)
    # p.beta_annual = p.beta_annual * 0.7
    #S = og_spec["S"]
    #T = og_spec["T"]
    #J = og_spec["J"]

    # Shape: T*S
    vars_cfg = {
        "labor_income_tax_noncompliance_rate": {"adjustments": ["T+S", "J"]},
        "capital_income_tax_noncompliance_rate": {"adjustments": ["T+S", "J"]},
        "replacement_rate_adjust": {"adjustments": ["T+S", "J"]},
        "rho": {"adjustments": ["T+S", "S"]},
        "eta": {"adjustments": ["T+S", "S", "J"]},
        "eta_RM": {"adjustments": ["T+S", "S", "J"]},
        "e": {"adjustments": ["T", "S", "J"]},
        "omega_SS": {"adjustments": ["S"]},
        "omega": {"adjustments": ["T+S", "S"]},
        "omega_S_preTP": {"adjustments": ["S"]},
        "imm_rates": {"adjustments": ["T+S", "S"]},
        "zeta": {"adjustments": ["S", "J"]},
        "g_n": {"adjustments": ["T+S"]},
        "etr_params": {"adjustments": ["T+S", "S", "PSEUDO_DIM1"]},
        "chi_n": {"adjustments": ["T+S", "S"]},
        "chi_b": {"adjustments": ["J"]},
        "tau_b": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "mtrx_params": {"adjustments": ["T+S", "S", "PSEUDO_DIM1"]},
        "mtry_params": {"adjustments": ["T+S", "S", "PSEUDO_DIM1"]},
        "frac_tax_payroll": {"adjustments": ["T+S"]},
        "retirement_age": {"adjustments": ["T+S"]},
        "Z": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "zeta_D": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "zeta_K": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "world_int_rate_annual": {"adjustments": ["T+S"]},
        "alpha_T": {"adjustments": ["T+S"]},
        "alpha_G": {"adjustments": ["T+S"]},
        "alpha_I": {"adjustments": ["T+S"]},
        "alpha_bs_T": {"adjustments": ["T+S"]},
        "alpha_bs_G": {"adjustments": ["T+S"]},
        "alpha_bs_I": {"adjustments": ["T+S"]},
        "g_RM": {"adjustments": ["T+S"]},
        "r_gov_scale": {"adjustments": ["T+S"]},
        "r_gov_shift": {"adjustments": ["T+S"]},
        "cit_rate": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "adjustment_factor_for_cit_receipts": {"adjustments": ["T+S"]},
        "inv_tax_credit": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "tau_c": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "delta_tau_annual": {"adjustments": ["T+S", "PSEUDO_DIM2"]},
        "h_wealth": {"adjustments": ["T+S"]},
        "m_wealth": {"adjustments": ["T+S"]},
        "p_wealth": {"adjustments": ["T+S"]},
        "tau_bq": {"adjustments": ["T+S"]},
        "tau_payroll": {"adjustments": ["T+S"]}
    }

    # for var in ["labor_income_tax_noncompliance_rate", "capital_income_tax_noncompliance_rate", "replacement_rate_adjust", "rho"]:
    for var in vars_cfg:
        proc_value = resample_method1(
            getattr(p, var), 
            og_spec = og_spec,
            adjustments = vars_cfg[var]["adjustments"],
            plot_cfg = {"run_plot": True, "param_name": var, "savedir": f"{save_dir}/params"})

        setattr(p, var, proc_value)

    p.nu = 0.1
    p.maxiter = 500

    p.starting_age = 2
    p.ending_age = og_spec["S"] + 2

    p.yr_contrib = None # not used ...
    p.avg_earn_num_years = 40 # using S = 80 as a reference
    p.PIA_maxpayment = 0.0 # in NZ, pension is a bit like ubi
    p.ubi_nom_65p = 24000
    p.ubi_nom_max = 99999
    p.ubi_growthadj = True
    p.retirement_age = (1 + p.retirement_age / og_spec["S"]).astype(int)


    p.zeta_K = 0.3 * ones(p.zeta_K.shape) # If zeta_K = 1: This makes foreign private capital flows fully absorb any gap between domestic saving and investment (original value: 0.1)
    p.initial_guess_TR_SS = 0.03 * ones(p.initial_guess_TR_SS.shape) # govt trasnfer (orig: 0.057)
    p.world_int_rate_annual = 0.04 * ones(p.world_int_rate_annual.shape) # world_int_rate_annual → set this to your desired exogenous world real interest rate (e.g., orig: 0.04).
    p.zeta_D = 0.5 * ones(p.zeta_D.shape) # how “open” you want the government bond market (orig: 0.4)
    p.initial_foreign_debt_ratio = 0.4 * ones(p.initial_foreign_debt_ratio.shape) # orig: 0.4
    p.beta_annual = 0.3 * ones(p.beta_annual.shape) # decrease -> less saving → smaller K → lower Y (orig: 0.96)
    p.Z = 0.15 * ones(p.Z.shape) # decrease -> scales Y down (orig: 1.0)
    p.alpha_G = 0.03 * ones(p.alpha_G.shape) # 0.5 * ones(p.alpha_G.shape) # increase -> raise demand (orig: 0.05)
    p.alpha_I = 0.0 * ones(p.alpha_I.shape) # increase -> raise demand (original value: 0)
    p.chi_n = p.chi_n * 1.2 # increase -> less hours → lower Y
    p.alpha_T = 0.18 * ones(p.alpha_T.shape) # orig value: 0.09
    p.initial_Kg_ratio = 0.3 # initial govertment  (orig: 0.0)
    p.debt_ratio_ss = 0.75 # debt ratio in steady state
    p.alpha_I = 0.15 * ones(p.alpha_I.shape) # govt infras investment (orig: 0.3)
    # p.etr_params = 1.15 * asarray(p.etr_params)
    #p.lambdas = asarray([[0.99999],
    #   [0.000001],
    #   [0.000001 ],
    #   [0.000001 ],
    #   [0.000001 ],
    #   [0.000001],
    #   [0.000001]])

    p.RC_SS = 0.003
    p.RC_TPI = 0.015
    #p.RC_TPI = 0.001
    #p.mindist_TPI = 0.0008

    # og_spec["chi_n"] = interpolate_partial(p.chi_n, 1, 0, S, kind="linear")
    # p.update_specifications(og_spec)
    # p.constant_demographics = True
    #p.frisch = p.frisch /2.0
    #p.delta_annual = p.delta_annual * 10.0
    #p.delta_g_annual = p.delta_g_annual * 10.0
    #p.delta_tau_annual = p.delta_tau_annual * 10.0
    #p.world_int_rate_annual = p.world_int_rate_annual * 10.0
    #p.g_y_annual = p.g_y_annual * 10.0
    for var in ["PSEUDO_DIM1", "PSEUDO_DIM2"]:
        og_spec.pop(var)

    p.update_specifications(og_spec)

    get_param(p, save_dir)
    
    return p