def multiarith_data_process(dataset):
    """
    Process MultiArith dataset
    
    Args:
        dataset: Raw MultiArith dataset
        
    Returns:
        list_data_dict: List of processed data dictionaries
    """
    list_data_dict = []
    for data in dataset:
        item = {"task": data["question"]}
        item["step"] = data.get("chain", "")
        # Support both 'answer' and 'final_ans' keys
        item["answer"] = data.get("answer", data.get("final_ans", ""))
        list_data_dict.append(item)

    return list_data_dict
