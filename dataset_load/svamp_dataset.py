from dataset_load.gsm8k_dataset import gsm_get_predict as get_predict


def svamp_data_process(dataset):
    """
    Process SVAMP dataset
    
    Args:
        dataset: Raw SVAMP dataset
        
    Returns:
        list_data_dict: List of processed data dictionaries
    """
    list_data_dict = []
    for data in dataset:
        task = data['Body'] + ' ' + data['Question']
        item = {"task": task}
        item["step"] = ''
        item["answer"] = data["Answer"]
        list_data_dict.append(item)

    return list_data_dict
