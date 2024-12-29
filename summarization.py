

import json
import os
from utils import get_output_file_path

def final_summarize_helper(
    output_file,
    vote_type = 1,
    null_type = 0,
): 
    # vote_type: 
    #   0 -> no voting; 
    #   1 -> entry is 1 if and only if all entries are 1

    # null_type: 
    #   0 -> return the original length [let empty entries to be zero]; 
    #   1 -> return the length without empty entries, i.e., length - empty_list_count

    if not os.path.exists(output_file):
        print(f"Error: File '{output_file}' not found.")
        exit()
    else:
        with open(output_file, "r") as f:
            ori_entries = json.load(f)
            print(f"Loaded '{output_file}' with {len(ori_entries)} entries.")

    
    def process_completion_list_by_voting(l, vote_type):
        if vote_type == 0: # no voting
            return l
        elif vote_type == 1: # entry is 1 if and only if all entries are 1
            return [0 if x <= 0.9999 else 1 for x in l]


    def get_len_entry(length, empty_list_count, null_type):
        # if null_type == 0 -> return the original length [let empty entries to be zero]; 
        # if null_type == 1 -> return the length without empty entries, i.e., length - empty_list_count
        assert length >= empty_list_count
        if null_type == 0:
            return length
        elif null_type == 1:
            return length - empty_list_count
        else:
            raise ValueError("null_type should be 0 or 1")


    def analyze_entry(ori_entries, selected_keynames = [], keyname = 'class'):
        if selected_keynames == []:
            entries = ori_entries
        else:
            # filter only the selected classes
            entries = {}
            if keyname == 'class':
                for key, value in ori_entries.items():
                    # if any class in value['class'] is in selected_classes, add it to entries
                    if any(c in selected_keynames for c in value['class']):
                        entries[key] = value

            elif keyname == 'model':
                for key, value in ori_entries.items():
                    if value['model'] in selected_keynames:
                        entries[key] = value

        length = len(entries)
        if length == 0:
            print(f"No entries found for {selected_keynames}")
            return
        
        completion_sum = 0
        empty_list_count = 0

        for key, entry in entries.items():
            completion_list = entry["completion_list_avg"]
            completion_list = process_completion_list_by_voting(completion_list, vote_type)

            # add the empty_list_count
            if len(completion_list) == 0:
                empty_list_count += 1
                    

            completion_sum += sum(completion_list) / len(completion_list) if len(completion_list) > 0 else 0

        
        empty_rate = empty_list_count / len(entries)

        all_num = get_len_entry(len(entries), empty_list_count, null_type)
        completion_avg = completion_sum / all_num

        selected_keynames = 'all' if selected_keynames == [] else selected_keynames
        print(f"completion average: {completion_avg} for {length} entries in class {selected_keynames}, empty rate: {empty_rate}, empty_list_count: {empty_list_count}, all_num: {all_num}")

        return completion_avg
    


    classes = ['human', 'animal', 'object', 'retrieval', 'creative', 
               'easy','hard', #  'medium', 
                # '2_events', '3_events', '4_events',
            ]
    print(f"Classes: {classes}")


    for c in classes:
        analyze_entry(ori_entries, [c])
        print('-----------------------------------')


    print("Final result:")
    analyze_entry(ori_entries)


def final_summarize(
    results_dir, 
    eval_model_type = 'gpt4o', 
    output_postfix = '',
    generative_model_name = 'all',
    vote_type = 1,
    null_type = 0,
):
    print(f'====================================== Final summarizing for {results_dir}, {generative_model_name} ======================================')
    print(f'Eval model type: {eval_model_type}\nOutput postfix: {output_postfix}\nGenerative model name: {generative_model_name}\nVote type: {vote_type}\nNull type: {null_type}\n')

    if generative_model_name == 'all':
        # all files has format f'modelname_gpt4o{output_postfix}.json', get the modelname
        for file in os.listdir(results_dir):
            # print(f'file: {file}')
            if file.endswith(f'_{eval_model_type}{output_postfix}.json'):
                tid = file.split('_')[0]
                print(f"====================================== Processing {tid} ======================================")
                output_file = get_output_file_path(results_dir, tid, eval_model_type, output_postfix)
                final_summarize_helper(output_file, vote_type, null_type)

    else:
        output_file = get_output_file_path(results_dir, generative_model_name, eval_model_type, output_postfix)
        final_summarize_helper(output_file, vote_type, null_type)


if __name__ == '__main__':
    results_dir = './full_results/'
    eval_model_type = 'gpt4o'
    output_postfix = '_final'
    generative_model_name = 'all'
    vote_type = 1
    null_type = 0
    final_summarize(results_dir, eval_model_type, output_postfix, generative_model_name, vote_type, null_type)
