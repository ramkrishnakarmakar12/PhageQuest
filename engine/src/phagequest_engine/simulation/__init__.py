"""Host-phage interaction simulation (spec section 03.3).

The spec's key finding: there is no maintained, general-purpose phage-dynamics
package anywhere, so this block is ours to define. Everything here runs in
Pyodide, which means the simulation block costs nothing per student.
"""
from .ode import MODELS, PARAM_META, simulate, parameter_sweep, one_step_growth
from .stochastic import gillespie_lytic, extinction_probability, poisson_plaques
from .plaque import plaque_growth, STATES
from .labmath import dilution_series, pfu_per_ml, titre_from_plates, moi

__all__ = [
    "MODELS", "PARAM_META", "simulate", "parameter_sweep", "one_step_growth",
    "gillespie_lytic", "extinction_probability", "poisson_plaques",
    "plaque_growth", "STATES",
    "dilution_series", "pfu_per_ml", "titre_from_plates", "moi",
]
