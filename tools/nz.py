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


    #og_spec["labor_income_tax_noncompliance_rate"] = interpolate_partial(p.labor_income_tax_noncompliance_rate, 0, T, S, kind="linear")
    #og_spec["capital_income_tax_noncompliance_rate"] = interpolate_partial(p.capital_income_tax_noncompliance_rate, 0, T, S, kind="linear")
    #og_spec["replacement_rate_adjust"] = interpolate_partial(p.replacement_rate_adjust, 0, T, S, kind="linear")
    #tmp_rho = interpolate_partial(p.rho, 0, T, S, kind="linear")
    #og_spec["rho"] = interpolate_partial(tmp_rho, 1, 0, S, kind="linear")
    
    #tmp_eta = interpolate_partial(p.eta, 0, T, S, kind="linear")
    #p.eta = interpolate_partial(tmp_eta, 1, 0, S, kind="linear")
    
    #tmp_eta_RM = interpolate_partial(p.eta_RM, 0, T, S, kind="linear")
    #p.eta_RM = interpolate_partial(tmp_eta_RM, 1, 0, S, kind="linear")

    # Shape: S
    #vars = {
    #    "e": {"scaler": False},
    #    "omega_SS": {"scaler": True}
    #}

    # for var in ["labor_income_tax_noncompliance_rate", "capital_income_tax_noncompliance_rate", "replacement_rate_adjust", "rho"]:
    #for var in vars:
    #    og_spec[var]  = resample_s(
    #        getattr(p, var), S, method="linear",
    #        apply_scaler=vars[var]["scaler"],
    #        plot_cfg = {
    #            "run_plot": True, 
    #            "param_name": var, 
    #            "savedir": f"{save_dir}/params"}
    #        )



    #p.e = interpolate_partial(p.e, 1, 0, S, kind="linear")
    #tmp_omega_SS = interpolate_partial(p.omega_SS, 0, 0, S, kind="linear")
    #tmp_omega_SS = tmp_omega_SS * (1.0 / tmp_omega_SS.sum())
    # og_spec["omega_SS"] = interpolate_partial(tmp_omega_SS, 0, 0, S, kind="linear")

    #tmp_imm_rates = interpolate_partial(p.imm_rates, 0, T, S, kind="linear")
    #og_spec["imm_rates"] = interpolate_partial(tmp_imm_rates, 1, 0, S, kind="linear")

    # og_spec["zeta"] = interpolate_partial(p.zeta, 0, 0, S, kind="linear")
    # og_spec["omega"]  = resample_2d(p.omega, T, S, method="linear")
    #tmp_omega = interpolate_partial(p.omega, 0, T, S, kind="linear")
    #tmp_omega = interpolate_partial(tmp_omega, 1, 0, S, kind="linear")
    #og_spec["omega"] = interpolate_partial(tmp_omega, 1, 0, S, kind="linear")
    # og_spec["omega_S_preTP"] = interpolate_partial(p.omega_S_preTP, 0, 0, S, kind="linear")
    #og_spec["g_n"] = interpolate_partial(p.g_n, 0, T, S, kind="linear")

    #tmp_etr_params = np.array(p.etr_params)
    #tmp_etr_params = interpolate_partial(tmp_etr_params, 0, T, S, kind="linear")
    #tmp_etr_params = interpolate_partial(tmp_etr_params, 1, 0, S, kind="linear")
    #og_spec["etr_params"] = tmp_etr_params.tolist()

    #tmp_mtrx_params = np.array(p.etr_params)
    #tmp_mtrx_params = interpolate_partial(tmp_mtrx_params, 0, T, S, kind="linear")
    #tmp_mtrx_params = interpolate_partial(tmp_mtrx_params, 1, 0, S, kind="linear")
    # og_spec["mtrx_params"] = tmp_mtrx_params.tolist()

    #tmp_chi_n = interpolate_partial(p.chi_n, 0, T, S, kind="linear")
    # p.chi_n = interpolate_partial(tmp_chi_n, 1, 0, S, kind="linear")
    #p.chi_n = interpolate_partial(tmp_chi_n, 1, 0, S, kind="linear")
    #p.chi_b = S * np.ones(p.chi_b.shape)
    #p.tau_b = interpolate_partial(p.tau_b, 0, T, S, kind="linear")

    #tmp_mtry_params = np.asarray(p.mtry_params)
    #tmp_mtry_params = interpolate_partial(tmp_mtry_params, 0, T, S, kind="linear")
    #p.mtry_params = interpolate_partial(tmp_mtry_params, 1, 0, S, kind="linear")

    #tmp_mtrx_params = np.asarray(p.mtrx_params)
    #tmp_mtrx_params = interpolate_partial(tmp_mtrx_params, 0, T, S, kind="linear")
    #p.mtrx_params = interpolate_partial(tmp_mtrx_params, 1, 0, S, kind="linear")
    #p.yr_contrib = int((p.yr_contrib / p.S) * og_spec["S"])
    #p.avg_earn_num_years = int((p.avg_earn_num_years / p.S) * og_spec["S"])

    p.yr_contrib = None # not used ...
    p.avg_earn_num_years = 40 # using S = 80 as a reference
    p.PIA_maxpayment = 0.0 # in NZ, pension is a bit like ubi
    p.ubi_nom_65p = 24000
    p.ubi_nom_max = 99999
    p.ubi_growthadj = True

    # p.maxiter = 50
    # p.debt_ratio_ss = 0.4
    # p.frac_tax_payroll = interpolate_partial(p.frac_tax_payroll, 0, T, S, kind="linear")
    # p.pension_system = "Defined Benefits"
    # p.nu = 0.01
    p.nu = 0.4
    p.maxiter = 500
    # p.SS_root_method = "lm" # essential change for SS optimization
    #p.FOC_root_method = "diagbroyden"
    #p.initial_guess_TR_SS = p.initial_guess_TR_SS * 30.0
    #p.initial_guess_r_SS = p.initial_guess_r_SS / 1000.0
    # p.initial_guess_r_SS = p.initial_guess_r_SS / 10.0
    #p.retirement_age = interpolate_partial(p.retirement_age, 0, T, S, kind="linear")
    p.retirement_age = (1 + p.retirement_age / og_spec["S"]).astype(int)

    p.starting_age = 2
    p.ending_age = og_spec["S"] + 2

    p.zeta_K = 0.5 * p.zeta_K # If zeta_K = 1: This makes foreign private capital flows fully absorb any gap between domestic saving and investment 
    p.initial_guess_TR_SS = 0.3 * p.initial_guess_TR_SS # govt trasnfer
    p.world_int_rate_annual = p.world_int_rate_annual / 2.0 # world_int_rate_annual → set this to your desired exogenous world real interest rate (e.g., 0.03–0.05).
    p.zeta_D = p.zeta_D * 1.5 # how “open” you want the government bond market:
    p.initial_foreign_debt_ratio = p.initial_foreign_debt_ratio * 1.5
    p.beta_annual = p.beta_annual / 3.0 # decrease -> less saving → smaller K → lower Y
    p.Z = p.Z / 5.0 # decrease -> scales Y down
    p.alpha_G = 3.0 * p.alpha_G # 0.5 * ones(p.alpha_G.shape) # increase -> raise demand (orig: 0.05)
    p.alpha_I = 0.3 * ones(p.alpha_I.shape) # increase -> raise demand (original value: 0)
    p.chi_n = p.chi_n * 1.2 # increase -> less hours → lower Y
    p.alpha_T = p.alpha_T * 2.0 
    p.initial_Kg_ratio = 0.3 # initial govertment 

    #p.lambdas = asarray([[0.99999],
    #   [0.000001],
    #   [0.000001 ],
    #   [0.000001 ],
    #   [0.000001 ],
    #   [0.000001],
    #   [0.000001]])

    # p.debt_ratio_ss = 0.4
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