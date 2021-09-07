
def generate_experiment_id(name,
                           seed=None, k=None, strategy=None, importance_measure=None,
                           recursive=None, riemann_samples=None):
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
