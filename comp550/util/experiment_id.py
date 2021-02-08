
def generate_experiment_id(name, seed, k, importance_measure, recursive):
    experiment_id = f"{name}_s-{seed}"
    if k >= 1:
        experiment_id += f"_k-{k}_m-{importance_measure[0]}_r-{int(recursive)}"
    return experiment_id
