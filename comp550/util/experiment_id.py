
def generate_experiment_id(name, seed, k=0, strategy='count', importance_measure=None, recursive=False):
    experiment_id = f"{name}_s-{seed}"
    if k > 0:
        assert isinstance(importance_measure, str)
        experiment_id += f"_k-{k}_y-{strategy[0]}_m-{importance_measure[0]}_r-{int(recursive)}"
    return experiment_id
