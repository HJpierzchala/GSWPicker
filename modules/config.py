"""
gsw_config.py
"""

from __future__ import annotations
from pathlib import Path
import tomllib

CFG_DFLT: dict[str, dict[str, object]] = {
    "PICKING": {
        "rareevent_t":           5,
        "shock_separation_len":  2,
        "shock_separation_len_hr": 10,
        "beta_weight":           0.7,
        "gamma_weight":          0.3,
        "sigma scaler":          3
    },
    "SLOPE_METHOD": {
        "Qyy_scaler":            0.4,
        "alpha_significance":    0.05,
    },
    "UI": {
        "shaking_length_var":    3,
        "window_size_var":       5,
    },
    "PREINSPECT": {
        "PSD_NFFT_freq":         5000,
    },
    "TIME": {
        "MJD_base_date":         "1858-11-17",
        "GPST_start":            "1980-01-06T00:00:00",
    },
    "CPUS": {
            "cpu_counter":      ''
        }
}


_DEFAULT_CFG = (
    Path(__file__).resolve().parent
        .parent
        / "config" / "parameters.toml"
)

def load_cfg(cfg_path: str | Path = _DEFAULT_CFG) -> dict:
    """
    Loads a *.toml* file into a Python dictionary.

    Parameters
    ----------
    cfg_path : str | Path, optional
        Path to *parameters.toml*.  Defaults to
        `<repo‑root>/config/parameters.toml`.

    Returns
    -------
    dict
        Parsed TOML content.
    """
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)

try:
    CFG = load_cfg()
except Exception as e:
    CFG = CFG_DFLT
