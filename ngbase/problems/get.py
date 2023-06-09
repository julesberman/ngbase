from ngbase.problems.ac import get_ac_eq
from ngbase.problems.burgers import get_burgers_eq
from ngbase.problems.bz import get_bz_eq
from ngbase.problems.kdv import get_kdv
from ngbase.problems.ks import get_ks_eq
from ngbase.problems.problem import Problem
from ngbase.problems.vlasov2s import get_vlasov2s_eq
from ngbase.problems.vlasovfix import get_vlasovfix_eq
from ngbase.problems.wave import get_2D_wave_eq_bc


def get_problem(problem_name) -> Problem:
    if problem_name == "kdv":
        problem = get_kdv()
    elif problem_name == "wavebc":
        problem = get_2D_wave_eq_bc()
    elif problem_name == "vlasovfix":
        problem = get_vlasovfix_eq()
    elif problem_name == "vlasov2s":
        problem = get_vlasov2s_eq()
    elif problem_name == "ac":
        problem = get_ac_eq()
    elif problem_name == "ks":
        problem = get_ks_eq()
    elif problem_name == "burgers":
        problem = get_burgers_eq()
    elif problem_name == "bz":
        problem = get_bz_eq()
    else:
        raise Exception(f"Unknown Problem: {problem_name}")

    problem.name = problem_name
    return problem
