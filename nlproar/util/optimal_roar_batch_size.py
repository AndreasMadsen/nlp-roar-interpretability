
def optimal_roar_batch_size(dataset_name, model_type, importance_measure_name, use_gpu):
    """ Utility function for batch-size used in computing importance measures

    Because the MIMIC datasets are very large, the gradient importance measure
    runs out of memory. To avoid this, this function reduces the batch size.

    Args:
        dataset_name: the name of the dataset, ie. dataset.name
        importance_measure_name: the full name of the importance measure,
            can be 'random', 'mutual-information', 'attention', 'gradient',
            or 'integrated-gradient'
        use_gpu: boolean indicating if a GPU is used

    Returns:
        int, the batch size
    """
    if model_type == 'roberta':
        if importance_measure_name == 'gradient':
            return 8
        return 48

    if dataset_name in ['mimic-a', 'mimic-d']:
        if importance_measure_name == 'gradient':
            return 8

        return 64
    else:
        return 256
