import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--prompts_dir', type=str, default='prompts')
parser.add_argument('--prompt_file_name', type=str, default='prompts_debug')

args = parser.parse_args()

json_file = os.path.join(args.prompts_dir, args.prompt_file_name + '.json')
txt_file = os.path.join(args.prompts_dir, args.prompt_file_name + '.txt')


with open(json_file, 'r') as f:
    data = json.load(f)

print(f'Loaded {json_file}')

prompt_list = []
for filename, entry in data.items():
    prompt_list.append(entry['prompt'])

with open(txt_file, 'w') as f:
    for idx, prompt in enumerate(prompt_list):
        f.write(prompt)
        if idx != len(prompt_list) - 1:
            f.write('\n')

print(f'Write to {txt_file}')
