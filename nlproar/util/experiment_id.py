
def generate_experiment_id(name,
                           seed=None, k=None, strategy=None, importance_measure=None,
                           recursive=None, riemann_samples=None):
    """Creates a standardized experiment name

    The format is
        {name}_s-{seed}_k-{k}_y-{strategy[0]}_y-{importance_measure[0]}_r-{int(recursive)}_rs-{riemann_samples}
    Note that parts are only added when not None.

    Args:
        name: str, the name of the experiment, this is usually the name of the task
        seed: int, the models initialization seed
        k: int, indicates the amount of information removed by ROAR
        strategy: 'count' or 'quantile', the ROAR removal strategy
        importance_measure: 'random', 'mutual-information', 'attention', 'gradient',
            or 'integrated-gradient'
        recursive: bool, indicates if Recursive ROAR is used
        riemann_samples: int, the amount of samples when computing integrated-gradient

    Returns:
        string, the experiment identifier
    """
    experiment_id = f"{name}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    if k == 0:
        return experiment_id

    if isinstance(k, int):
        experiment_id += f"_k-{k}"
    if isinstance(strategy, str):
        experiment_id += f"_y-{strategy[0]}"
    if isinstance(importance_measure, str):
        experiment_id += f"_m-{importance_measure[0]}"
    if isinstance(recursive, bool):
        experiment_id += f"_r-{int(recursive)}"
    if isinstance(riemann_samples, int):
        riemann_samples = riemann_samples if importance_measure == 'integrated-gradient' else 0
        experiment_id += f"_rs-{riemann_samples}"

    return experiment_id
