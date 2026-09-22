"""Canonical naming for the existing best-improvement local search.

No search implementation lives here.  The public symbol is a direct alias to
the historical function, whose name is retained for backward compatibility.
"""

from pa_moap_rl.solvers.local_search_solver import local_search_best_improvement


one_point_best_improvement_local_search = local_search_best_improvement


__all__ = ["one_point_best_improvement_local_search"]
