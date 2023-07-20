import json
from tqdm import tqdm


def get_golden_paragraph(input_file, output_file):
    """
    The function `get_golden_paragraph` takes an input file containing JSON data, extracts relevant
    information (selected supporting facts for each data entry) from the data, and saves the 
    extracted information to an output file in JSON format.
    
    :param input_file: The input_file parameter is the path to the JSON file containing the data. This
    file should contain a list of dictionaries, where each dictionary represents a piece of information.
    Each dictionary should have the following keys:
    :param output_file: The `output_file` parameter is the file path where the output will be saved. It
    should be a JSON file
    """
    data = json.load(open(input_file, "r"))
    sp_dict = {}
    for info in tqdm(data):
        sup = info['supporting_facts']
        contexts = info['context']
        get_id = info['_id']
        sp_dict[get_id] = []
        for context_idx, context in enumerate(contexts):
            title, sentences = context
            for sentence_idx, sentence in enumerate(sentences):
                if [title, sentence_idx] in sup:
                    if get_id in sp_dict:
                        sp_dict[get_id].append(context_idx)
        sp_dict[get_id] = list(set(sp_dict[get_id]))
    print(output_file)
    json.dump(sp_dict, open(output_file, "w"))


if __name__ == '__main__':
    input_files = ["../data/hotpot_data/hotpot_train_labeled_data_v3.json",
                   "../data/hotpot_data/hotpot_dev_labeled_data_v3.json"]
    output_files = ["../data/hotpot_data/train_golden.json",
                    "../data/hotpot_data/dev_golden.json"]
    for input_file, output_file in zip(input_files, output_files):
        get_golden_paragraph(input_file, output_file)

