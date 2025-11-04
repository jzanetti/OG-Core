"""
OG-Core Parameter Scaler — Dash app

What this does
- Loads an OG-Core parameter JSON (or uses built-in sample defaults).
- Lets you scale selected parameters up/down with sliders (multiplicative factors).
- Previews the resulting values and lets you download a modified JSON.

How to run
1) Create/activate a virtual env.
2) pip install dash==2.* pandas numpy
   # If you have OG-Core installed, also: pip install ogcore
3) Save this file as app.py
4) python app.py
5) Open http://127.0.0.1:8050 in your browser.

Notes
- If the OG-Core package is available, the app will try to locate its default parameter file automatically.
- You can also upload any OG-Core-style parameters JSON (e.g. ogcore_default_parameters.json) via the UI.
- Scalar parameters are scaled directly; arrays/lists are scaled element-wise.
- Use the Download button to export the modified parameters.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, dash_table, ctx, ALL
import run_ogcore_nz

# -------------------------------
# Try to locate OG-Core defaults
# -------------------------------

def find_ogcore_default_params() -> Path | None:
    """Try to find ogcore's default parameter JSON on the local system."""
    try:
        import ogcore  # type: ignore
        # Common locations across versions; adjust if your install differs
        candidates = [
            Path(ogcore.__file__).parent / "ogcore_default_parameters.json",
            Path(ogcore.__file__).parent / "ogcore_default_parameters_base.json",
            Path(ogcore.__file__).parent / "ogcore_default_parameters.yaml",
        ]
        for p in candidates:
            if p.exists():
                return p
    except Exception:
        pass
    return None

# -------------------------------
# Fallback sample parameters
# -------------------------------

SAMPLE_PARAMS: Dict[str, Any] = {
    # Typical OG-Core-style parameters (names here are indicative)
    "beta": 0.96,
    "sigma": 2.0,
    "chi_n": [0.6] * 80,  # disutility of labor by age (example vector)
    "alpha": 0.35,        # capital share
    "delta": 0.06,        # depreciation
    "Z": 1.0,             # TFP level
    "tau_c": 0.05,        # consumption tax
    "tau_payroll": 0.12,  # payroll tax
    "g_y": 0.02,          # tech growth
    "rho": 0.04,          # interest rate guess
}

DEFAULT_SCALABLE = [
    "beta",
    "sigma",
    "chi_n",
    "alpha",
    "delta",
    "Z",
    "tau_c",
    "tau_payroll",
    "g_y",
]

# -------------------------------
# Utility helpers
# -------------------------------

def load_params_from_file(p: Path) -> Dict[str, Any]:
    if p.suffix.lower() == ".json":
        with open(p, "r") as f:
            return json.load(f)
    elif p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "YAML file provided but PyYAML not installed. Run: pip install pyyaml"
            ) from e
        with open(p, "r") as f:
            return yaml.safe_load(f)
    else:
        raise ValueError("Unsupported parameter file type (use .json, .yaml, or .yml)")


def flatten_for_preview(value: Any, max_len: int = 6) -> str:
    """Compact preview string for scalar/list/nd arrays."""
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value)
        flat = arr.flatten()
        n = min(max_len, flat.size)
        prefix = ", ".join(f"{x:.4g}" for x in flat[:n])
        suffix = "" if flat.size <= n else ", …"
        return f"[{prefix}{suffix}] (shape={list(arr.shape)})"
    if isinstance(value, dict):
        return "{...}"  # nested; OG-Core doesn't usually nest numerics deeply
    return str(value)


def apply_scale(value: Any, factor: float) -> Any:
    if isinstance(value, (int, float)):
        return value * factor
    if isinstance(value, (list, tuple, np.ndarray)):
        arr = np.array(value, dtype=float) * factor
        # Return same container type for JSON serialization
        return arr.tolist()
    # Non-numeric -> return unchanged
    return value


def build_preview_table(params: Dict[str, Any], factors: Dict[str, float], selected: list[str]) -> pd.DataFrame:
    rows = []
    for k in selected:
        base = params.get(k, "<missing>")
        f = float(factors.get(k, 1.0))
        new_val = apply_scale(base, f) if base != "<missing>" else "<missing>"
        rows.append(
            {
                "parameter": k,
                "factor": f,
                "base": flatten_for_preview(base),
                "new": flatten_for_preview(new_val),
            }
        )
    return pd.DataFrame(rows)


def scaled_params_dict(params: Dict[str, Any], factors: Dict[str, float], selected: list[str]) -> Dict[str, Any]:
    out = {**params}
    for k in selected:
        if k in out:
            try:
                out[k] = apply_scale(out[k], float(factors.get(k, 1.0)))
            except Exception:
                # Leave unmodified on error
                pass
    return out

# -------------------------------
# Dash app
# -------------------------------

app = Dash(__name__)
app.title = "OG-Core Parameter Scaler"

# Try to auto-load OG-Core default
autopath = find_ogcore_default_params()
initial_params = SAMPLE_PARAMS
source_label = "Built-in sample defaults"
if autopath is not None:
    try:
        initial_params = load_params_from_file(autopath)
        source_label = f"Loaded from {autopath}"
    except Exception:
        pass

initial_selected = [k for k in DEFAULT_SCALABLE if k in initial_params]
initial_factors = {k: 1.0 for k in initial_selected}

app.layout = html.Div(
    className="container",
    children=[
        html.H2("OG-Core Parameter Scaler"),
        html.Div(
            f"Parameter source: {source_label}",
            style={"fontStyle": "italic", "marginBottom": "8px"},
        ),
        html.Details(
            open=False,
            children=[
                html.Summary("Upload a parameter file (.json/.yaml)"),
                dcc.Upload(
                    id="param-upload",
                    children=html.Div([
                        "Drag and drop or ", html.A("select a file")
                    ]),
                    multiple=False,
                    style={
                        "width": "100%",
                        "height": "90px",
                        "lineHeight": "90px",
                        "borderWidth": "1px",
                        "borderStyle": "dashed",
                        "borderRadius": "6px",
                        "textAlign": "center",
                        "margin": "10px 0",
                    },
                ),
            ],
        ),
        html.Div(
            [
                html.Label("Choose parameters to scale"),
                dcc.Dropdown(
                    id="param-choose",
                    options=[{"label": k, "value": k} for k in sorted(initial_params.keys())],
                    value=initial_selected,
                    multi=True,
                    placeholder="Select parameters…",
                ),
            ],
            style={"margin": "12px 0"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Global factor (multiplies all sliders)"),
                        dcc.Slider(
                            id="global-factor",
                            min=0.25,
                            max=2.0,
                            step=0.01,
                            value=1.0,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(id="global-factor-val", style={"marginTop": "4px"}),
                    ],
                    style={"marginBottom": "20px"},
                ),
                html.Div(id="sliders-panel"),
            ]
        ),
        html.Hr(),
        html.Div(
            [
                html.Button("Reset factors", id="reset-factors", n_clicks=0),
                html.Button(
                    "Download modified JSON",
                    id="download-json-btn",
                    n_clicks=0,
                    style={"marginLeft": "8px"},
                ),
                dcc.Download(id="download-json"),
            ],
            style={"margin": "10px 0"},
        ),
        html.Div(
            [
                html.H4("Preview: base vs new"),
                dash_table.DataTable(
                    id="preview-table",
                    columns=[
                        {"name": "parameter", "id": "parameter"},
                        {"name": "factor", "id": "factor"},
                        {"name": "base", "id": "base"},
                        {"name": "new", "id": "new"},
                    ],
                    data=[],
                    style_table={"overflowX": "auto"},
                    style_cell={"fontFamily": "monospace", "fontSize": "13px"},
                ),
            ]
        ),
        # Hidden stores
        dcc.Store(id="params-store", data=initial_params),
        dcc.Store(id="factors-store", data=initial_factors),
    ],
)

# -------------------------------
# Callbacks
# -------------------------------

@app.callback(
    Output("global-factor-val", "children"),
    Input("global-factor", "value"),
)
def show_global_val(gf):
    return f"Global factor = {gf:.2f} (applied multiplicatively to each slider)"


@app.callback(
    Output("sliders-panel", "children"),
    Input("param-choose", "value"),
    State("factors-store", "data"),
)
def render_sliders(selected: list[str], factors_state: Dict[str, float]):
    selected = selected or []
    children = []
    for k in selected:
        val = float(factors_state.get(k, 1.0))
        children.append(
            html.Div(
                [
                    html.Label(k, style={"fontWeight": 600}),
                    dcc.Slider(
                        id={"type": "param-slider", "param": k},
                        min=0.25,
                        max=2.0,
                        step=0.01,
                        value=val,
                        marks=None,
                        tooltip={"placement": "bottom", "always_visible": False},
                    ),
                    html.Div(id={"type": "param-slider-out", "param": k}, style={"marginBottom": "8px"}),
                ],
                style={"marginBottom": "16px"},
            )
        )
    return children


@app.callback(
    Output({"type": "param-slider-out", "param": ALL}, "children"),
    Input({"type": "param-slider", "param": ALL}, "value"),
    State({"type": "param-slider", "param": ALL}, "id"),
)
def show_each_slider_val(values, ids):
    outs = []
    for v, i in zip(values or [], ids or []):
        outs.append(f"factor = {float(v):.2f}")
    return outs


@app.callback(
    Output("factors-store", "data"),
    Input({"type": "param-slider", "param": ALL}, "value"),
    State({"type": "param-slider", "param": ALL}, "id"),
    State("global-factor", "value"),
    prevent_initial_call=True,
)
def update_factors_from_sliders(values, ids, global_factor):
    factors = {}
    for v, i in zip(values or [], ids or []):
        name = i.get("param")
        factors[name] = float(v) * float(global_factor or 1.0)
    return factors


@app.callback(
    Output("preview-table", "data"),
    Input("factors-store", "data"),
    State("params-store", "data"),
    State("param-choose", "value"),
)
def update_preview(factors, params, selected):
    df = build_preview_table(params, factors or {}, selected or [])
    return df.to_dict("records")


@app.callback(
    Output("params-store", "data"),
    Output("param-choose", "options"),
    Input("param-upload", "contents"),
    State("param-upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if not contents:
        raise RuntimeError("No contents")
    import base64
    import io

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    if filename.lower().endswith(".json"):
        params = json.load(io.StringIO(decoded.decode("utf-8")))
    elif filename.lower().endswith((".yaml", ".yml")):
        import yaml  # type: ignore
        params = yaml.safe_load(io.StringIO(decoded.decode("utf-8")))
    else:
        raise ValueError("Please upload a .json, .yaml, or .yml file")

    options = [{"label": k, "value": k} for k in sorted(params.keys())]
    return params, options


@app.callback(
    Output("download-json", "data"),
    Input("download-json-btn", "n_clicks"),
    State("params-store", "data"),
    State("factors-store", "data"),
    State("param-choose", "value"),
    prevent_initial_call=True,
)
def download_json(n_clicks, params, factors, selected):
    if not n_clicks:
        return dash.no_update  # type: ignore
    newp = scaled_params_dict(params, factors or {}, selected or [])
    payload = json.dumps(newp, indent=2)
    return dict(content=payload, filename="ogcore_parameters_scaled.json")


@app.callback(
    Output("factors-store", "data"),
    Input("reset-factors", "n_clicks"),
    State("param-choose", "value"),
    prevent_initial_call=True,
)
def reset_factors(n, selected):
    return {k: 1.0 for k in (selected or [])}


if __name__ == "__main__":
    app.run(debug=True)
