
from tools import get_default_params
from copy import deepcopy
from numpy import random as np_random
import numpy as np
import pandas as pd
import os
from ogcore.constants import REFORM_DIR, BASELINE_DIR
from ogcore.parameters import Specifications
from dill import dump as dill_dump
from ogcore.execute import runner
from shutil import rmtree

def run_eval_wrapper(save_dir, n: int = 100):

    for _ in range(n):

        p = obtain_base_params(save_dir, run_pertn = True, num_workers = 1)

        if p is not None:
            try:
                eval_results = runner(p, time_path=False, client=None, run_eval = True)
            except RuntimeError:
                pass
            
            if eval_results["dist"].values[0] < p.mindist_SS and eval_results["RC"].values[0] < p.RC_SS and abs(eval_results["Gss"].values[0]) <= 1e-3:
                dill_dump(p, open(f"{save_dir}/optimal_params.pickle", "wb" ) )
                rmtree(f"{save_dir}/SS")
                rmtree(f"{save_dir}/TPI")
                return
    
    raise Exception("not able to find the optimal params")


def obtain_base_params(work_dir, run_pertn = False, num_workers = 1):

    p = Specifications(
        baseline=True,
        num_workers=num_workers,
        baseline_dir=work_dir,
        output_base=work_dir,
    )
    return get_nz_params(p, work_dir, run_pertn=run_pertn)


def obtain_reform_params(p_base, save_dir):

    p_reform = deepcopy(p_base)

    reform_dir = os.path.join(save_dir, REFORM_DIR)
    p_reform.output_base = reform_dir
    p_reform.baseline = False
    # p_reform.alpha_I = 0.5 * p_base.alpha_I
    p_reform.debt_ratio_ss = 1.5 * p_base.debt_ratio_ss

    return p_reform


def get_base_params():
    ###############################################################
    #  DEMOGRAPHICS + TAX PARAMETERS FOR OG-Core (NZ, S=8 VERSION)
    ###############################################################

    T = 320  # Number of transition periods; long horizon ensures the economy reaches steady state (OG-Core requirement)
    S = 8    # Number of model age groups (compressed from 80 → 8 to save computation)
    J = 7    # Number of ability (productivity) types

    # --------------------
    # BASE DEMOGRAPHICS
    # --------------------
    # Base-year population shares across the 8 age groups (must sum to 1).
    # OG-Core: omega_S_preTP = distribution of population before the model starts.
    omega_base = np.array([0.13, 0.140, 0.149, 0.150, 0.145, 0.117, 0.064, 0.025])
    omega_base = omega_base * (1.0 / omega_base.sum())
    # Base immigration rates by age group.
    # NZ migration is concentrated among younger working ages.
    imm_base = np.array([0.012, 0.018, 0.025, 0.02, 0.007, 0.0002, 0.0001, 0.000])

    # Age-specific mortality rates; increase with age (OG-Core: rho).
    rho_vec = np.array([0.002, 0.003, 0.005, 0.009, 0.016, 0.032, 0.070, 0.150])

    # Time path for population growth: higher early, then stable.
    # OG-Core: g_n is population growth over the transition.
    g_n_path = np.array([0.010] * 40 + [0.008] * (T - 40))

    # Allocate full demographic arrays (OG-Core needs t = -S+1 … T).
    omega_full = np.zeros((T + S, S))
    imm_rates_full = np.zeros((T + S, S))

    # Pre-transition demographic state: use base-year distributions.
    for t in range(S):
        omega_full[t, :] = omega_base       # Copy base population shares
        imm_rates_full[t, :] = imm_base     # Copy base immigration pattern

    # Target steady-state age distribution (OG-Core: omega_SS).
    # omega_ss = np.array([0.143, 0.138, 0.133, 0.126, 0.115, 0.098, 0.076, 0.171])
    omega_ss = np.array([0.145, 0.140, 0.149, 0.150, 0.145, 0.127, 0.074, 0.065])
    current_omega = omega_base.copy()

    # --------------------
    # BUILD POPULATION PATH
    # --------------------
    for t in range(T):
        current_omega = current_omega * (1 + g_n_path[t])  # Apply population growth
        current_omega += imm_base * g_n_path[t] * 10       # Add immigration (scaled to match NZ flows)

        survived = np.zeros(S)
        survived[1:] = current_omega[:-1] * (1 - rho_vec[:-1])  # Survival & aging forward
        current_omega = survived

        current_omega /= current_omega.sum()                     # Normalise to sum to 1

        current_omega = 0.95 * current_omega + 0.05 * omega_ss   # Smooth adjustment toward steady state

        omega_full[S + t, :] = current_omega
        imm_rates_full[S + t, :] = imm_base                      # Keep immigration age pattern fixed


    ###############################################################
    #  TAX PARAMETERS (DEP FORM) — (10 × S × 12)
    ###############################################################
    # OG-Core: Tax functions *always* require:
    # shape = (10 parameters, S ages, 12 lifetime income groups)
    # even if J ≠ 12 (this is a separate classification used only for taxes).

    # --------------------------
    # 1. EFFECTIVE TAX RATE (ETR)
    # --------------------------
    etr_params = np.zeros((10, S, 12))  # (10 × 8 × 12)

    alpha_etr = np.array([1.00, 0.80, 0.65, 0.50, 0.40,
                        0.30, 0.20, 0.15, 0.10, 0.05])  # Decreasing weights (OG-Core standard)

    for p in range(10):
        for s in range(S):
            for j in range(12):
                base_etr = 0.12 + 0.02*s + 0.025*j    # NZ effective tax rises with age & income group
                etr_params[p, s, j] = alpha_etr[p] * base_etr


    # ------------------------------
    # 2. LABOR MARGINAL TAX RATE (MTRx)
    # ------------------------------
    # Smooth approximation of NZ PAYE marginal brackets (10.5% → 39%).

    mtrx_params = np.zeros((10, S, 12))

    alpha_mtrx = np.array([1.00, 0.90, 0.75, 0.60, 0.50,
                        0.40, 0.30, 0.20, 0.12, 0.06])

    for p in range(10):
        for s in range(S):
            for j in range(12):
                base_mtr = 0.13 + 0.03*s + 0.035*j    # Rising pattern: age & income group
                base_mtr = min(base_mtr, 0.39)        # Upper bound = highest NZ MTR
                mtrx_params[p, s, j] = alpha_mtrx[p] * base_mtr


    # --------------------------------
    # 3. CAPITAL MARGINAL TAX RATE (MTRy)
    # --------------------------------
    # NZ capital taxes are low: PIE cap ~0.39 → effective MTR ~10%–39%.

    mtry_params = np.zeros((10, S, 12))

    alpha_mtry = np.array([1.00, 0.85, 0.70, 0.55, 0.45,
                        0.35, 0.25, 0.18, 0.12, 0.06])

    for p in range(10):
        for s in range(S):
            for j in range(12):
                base_mtr = 0.10 + 0.015*s + 0.02*j     # NZ capital-income pattern
                base_mtr = min(base_mtr, 0.39)         # Effective max (NZ realistic)
                mtry_params[p, s, j] = alpha_mtry[p] * base_mtr


    ###############################################################
    #  FULL OG-Core SPECIFICATION DICTIONARY
    ###############################################################
    og_spec = {

        # ============================================================
        # 1. BASIC MODEL STRUCTURE
        # ============================================================
        'S': S,                     # Number of ages
        'J': 7,                     # Number of ability types
        'T': T,                     # Transition horizon
        'M': 1,                     # One production industry
        'I': 1,                     # One consumption good
        'starting_age': 21,         # Real-world age mapped to model age 1
        'ending_age': 92,           # Maximum modelled age
        'constant_demographics': False,  # Use full demographic transition
        'ltilde': 1.0,              # Time endowment per period
        
        # ============================================================
        # 2. DEMOGRAPHICS
        # ============================================================
        'omega': omega_full.tolist(),           # Full time path of population shares
        'omega_SS': omega_ss.tolist(),          # Steady-state age distribution
        'omega_S_preTP': omega_base.tolist(),   # Pre-transition population shares
        'g_n': g_n_path.tolist(),               # Population growth over transition
        'g_n_ss': 0.008,                        # Steady-state population growth
        'imm_rates': imm_rates_full.tolist(),   # Immigration rates over time
        'rho': [rho_vec.tolist()],              # Mortality rates

        # ============================================================
        # 3. ABILITY, LABOR SUPPLY, PRODUCTIVITY
        # ============================================================
        'lambdas': [0.25,0.25,0.20,0.15,0.10,0.04,0.01],  # Share of population in each ability group

        # Age-ability productivity matrix (e[s][j])
        'e': [
            [0.4,0.6,0.8,1.0,1.2,1.4,1.6],
            [0.6,0.8,1.0,1.2,1.4,1.6,1.8],
            [0.8,1.0,1.2,1.4,1.6,1.8,2.0],
            [0.9,1.1,1.3,1.5,1.7,1.9,2.1],
            [0.8,1.0,1.2,1.4,1.6,1.8,2.0],
            [0.6,0.8,1.0,1.2,1.4,1.6,1.8],
            [0.4,0.6,0.8,1.0,1.2,1.4,1.6],
            [0.2,0.4,0.6,0.8,1.0,1.2,1.4]
        ],

        # ============================================================
        # 4. PREFERENCES
        # ============================================================
        'frisch': 0.6,                # Frisch labor supply elasticity
        'sigma': 2.0,                 # CRRA (risk aversion)
        'beta_annual': [0.97] * T,    # Annual discount factor
        'chi_n': [30.0] * T,          # Weight on disutility of labor (increase -> smaller RC)
        'chi_b': [0.1] * T,           # No bequest motive

        # ============================================================
        # 5. TECHNOLOGY
        # ============================================================
        'g_y_annual': 0.02,          # Tech (labor-augmenting) growth rate (very sensitive)

        # ============================================================
        # 5. CAPITAL
        # ============================================================
        'gamma': [0.4] * T,          # Private capital share
        'gamma_g': [0.1] * T,        # Public capital share
        'epsilon': [1.0] * T,         # Elasticity of substitution (Cobb-Douglas)
        'delta_annual': 0.07,         # Private capital depreciation
        'delta_g_annual': 0.04,       # Public capital depreciation
        'Z': [[1.0]],                 # TFP level
        'io_matrix': [[1.0]],         # 1-good, 1-sector IO structure

        # ============================================================
        # 6. SMALL OPEN ECONOMY
        # ============================================================
        'world_int_rate_annual': [0.025] * T,  # Exogenous world interest rate (NZ-SOE assumption)
        'initial_foreign_debt_ratio': 0.4,   # Foreign-held share of gov't debt
        'zeta_D': [0.5] * T,                 # Foreign purchase share of new debt
        'zeta_K': [0.8] * T,                 # Foreign supply of excess capital

        # ============================================================
        # 7. GOVERNMENT BUDGET
        # ============================================================
        'debt_ratio_ss': 0.45,                # Target steady-state debt/GDP
        'initial_debt_ratio': 0.35,          # Current debt/GDP
        'initial_Kg_ratio': 0.3,             # Public capital/GDP baseline
        'tG1': 30,                           # Year when fiscal closure begins
        'tG2': 60,                           # Year when budget reaches SS rule
        'rho_G': 0.4,                        # Speed of budget adjustment
        'alpha_T': [0.194] * T,               # Transfers/GDP ratio (19.4%)
        'alpha_G': [0.209] * T,               # Government consumption/GDP (20.9%)
        'alpha_I': [0.048] * T,               # Public investment/GDP (4.8%)
        'r_gov_scale': [1.0] * T,            # Scaling of gov interest rate
        'r_gov_shift': [0.0] * T,            # Shift to gov interest rate
        'alpha_RM_1': 0.0,                   # No remittances
        'alpha_RM_T': 0.0,
        'g_RM': [0.0],

        # ============================================================
        # 8. TAX SYSTEM
        # ============================================================
        'cit_rate': [[0.28]],                # NZ corporate income tax rate
        'c_corp_share_of_assets': 0.7,       # Capital share in C-corps
        'adjustment_factor_for_cit_receipts': [0.85],  # Effective CIT adjustment
        'inv_tax_credit': [[0.0]],           # No investment tax credits in NZ
        'tau_c': [[0.15]],                   # GST = 15%
        'delta_tau_annual': [[0.07]],        # Depreciation for tax purposes
        'm_wealth': [1.0],                   # Wealth tax parameters → no wealth tax
        'p_wealth': [1.0],
        'tau_bq': [0.0],                     # No estate tax
        'tau_payroll': [0.0] * T,            # NZ has no payroll tax (KiwiSaver is voluntary)
        'constant_rates': True,
        'zero_taxes': False,
        'analytical_mtrs': False,
        'age_specific': False,

        'etr_params': etr_params.tolist(),   # Effective average tax rate tensor
        'mtrx_params': mtrx_params.tolist(), # Marginal tax on labor income
        'mtry_params': mtry_params.tolist(), # Marginal tax on capital income

        'mean_income_data': 75000,           # Mean taxable income in NZ
        'labor_income_tax_noncompliance_rate': [[0.0]], # NZ compliance high
        'capital_income_tax_noncompliance_rate': [[0.0]],
        'frac_tax_payroll': [0.0] * T,       # No payroll taxes → set to zero

        # ============================================================
        # 9. PENSION SYSTEM — NZ Super (treated as transfers)
        # ============================================================
        # 'pension_system': 'none',
        # 'avg_earn_num_years': 0,
        'retirement_age': [65] * T,          # NZ Super age
        'tau_p': 0.0,                        # No payroll-financed public pension
        'indR': 0.0,
        'k_ret': 1.0,
        'alpha_db': 0.0,
        'vpoint': 0.0,
        'yr_contrib': 0,
        'AIME_bkt_1': 0,
        'AIME_bkt_2': 0,
        'PIA_rate_bkt_1': 0.0,               # All US Social Security structures disabled
        'PIA_rate_bkt_2': 0.0,
        'PIA_rate_bkt_3': 0.0,
        'PIA_maxpayment': 0.0,
        'PIA_minpayment': 0.0,
        'replacement_rate_adjust': [[1.0]],  # Placeholder (not used because US system off)

        'ubi_growthadj': False,              # No UBI
        'ubi_nom_017': 0,
        'ubi_nom_1864': 0,
        'ubi_nom_65p': 0,
        'ubi_nom_max': 0,

        # ============================================================
        # 10. TRANSFERS / BEQUESTS
        # ============================================================
        'eta': [[1.0/7]*7 for _ in range(8)],  # Transfers distributed equally across ability types
        'eta_RM': [[0.0]*7 for _ in range(8)], # No remittances
        'zeta': [[0.1]*8 for _ in range(8)],   # No bequests
        'use_zeta': False,

        # ============================================================
        # 11. SOLVER SETTINGS
        # ============================================================
        'nu': 0.1,                 # Relaxation parameter for TPI stability
        'maxiter': 500,            # Max iterations in TPI solve
        # 'mindist_SS': 1e-13,        # SS convergence tolerance
        #'mindist_TPI': 1e-6,       # TPI convergence tolerance
        #'RC_SS': 1e-6,             # Resource constraint tolerance (SS)
        #'RC_TPI': 1e-5,            # Resource constraint tolerance (TPI)
        # 'budget_balance': False,   # Government need not balance budget each year

        # ============================================================
        # 12. INITIAL GUESSES
        # ============================================================
        'initial_guess_r_SS': 0.02,      # Initial guess for steady-state interest rate
        'initial_guess_TR_SS': 0.2,      # Initial guess for transfers
        'initial_guess_factor_SS': 1.0,  # Initial scaling factor
        'reform_use_baseline_solution': True,  # Reforms start from baseline SS

        # ============================================================
        # 13. ADMIN
        # ============================================================
        'start_year': 2025,              # Model starting calendar year

        # ============================================================
        # 14. MISSING
        # ============================================================
        'alpha_c': [0.1] # Share parameters for each good in the composite consumption good.
    }

    return og_spec


def get_nz_params(p, work_dir, run_pertn = False, bdy_limit = False):
    
    if run_pertn:
        nz_spec = get_base_params()

        default_p = get_default_params()    
        nz_spec_orig = deepcopy(nz_spec)
        all_vars = ["ltilde", "g_n_ss", "imm_rates", "sigma", "beta_annual", "chi_n", "world_int_rate_annual", "zeta_K", 
                    "debt_ratio_ss", "alpha_T", "alpha_G", "alpha_I", "alpha_c", "zeta_K",
                    "initial_guess_r_SS", "initial_guess_TR_SS", "initial_guess_factor_SS"]
        shocks = np_random.uniform(-0.01, 0.01, size=len(all_vars))
        param_records = {}
        for j, key in enumerate(all_vars):
            data_range = default_p[key]["validators"]["range"]
            data_range_min = data_range["min"]
            data_range_max = data_range["max"]
            try:
                org_value = nz_spec_orig[key] 
                upd_value = org_value * (1 + shocks[j])
                if bdy_limit:
                    if proc_value > data_range_max:
                        proc_value = data_range_max
                    if proc_value < data_range_min:
                        proc_value = data_range_min
                nz_spec[key] = upd_value
                org_value_mean = round(np.mean(org_value), 3)
                upd_value_mean = round(np.mean(upd_value), 3)
            except TypeError:
                org_value = np.asarray(nz_spec_orig[key])
                upd_value = org_value * (1 + shocks[j])
                if bdy_limit:
                    upd_value = np.clip(upd_value, data_range_min, data_range_max)
                org_value_mean = round(np.mean(org_value), 3)
                upd_value_mean = round(np.mean(upd_value), 3)
                nz_spec[key] = upd_value.tolist()

            param_records[key] = f"{org_value_mean} -> {upd_value_mean} ({round(shocks[j]*100, 1)}%)"

        csv_path = f"{work_dir}/results_param_updates.csv"
        df_row = pd.DataFrame([param_records])

        if not os.path.exists(csv_path):
            df_row.to_csv(csv_path, index=False)
        else:
            df_row.to_csv(csv_path, mode='a', header=False, index=False)
    
    p.update_specifications(nz_spec, raise_errors = False)

    return p