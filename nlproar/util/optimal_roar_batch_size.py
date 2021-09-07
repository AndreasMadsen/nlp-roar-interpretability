
def optimal_roar_batch_size(dataset_name, importance_measure_name, use_gpu):
    if dataset_name in ['mimic-a', 'mimic-d']:
        return 8 if importance_measure_name == 'gradient' or not use_gpu else 64
    else:
        return None if not use_gpu else 256
