import os

import re
import json
import copy
import torch
import warnings
from PIL import Image
import numpy as np
import time
import argparse
import requests
import time
import shutil
import random

from utils import load_video, load_video_by_opencv2, get_output_file_path
from summarization import final_summarize

# Constants
TARGET_WIDTH = 512 # videos will be resized to 512x320 for evaluation
TARGET_HEIGHT = 320
DEFAULT_IMAGE_TOKEN = "<image>"


def add_other_keys_in_entry(result, entry, add_conflict=1):
    existing_keys = result.keys()
    entry_keys = entry.keys()
    for key in entry_keys:
        if key not in existing_keys:
            result[key] = entry[key]
        else:
            # if add_conflict is 0, skip the key
            if add_conflict == 1 and key != "event_list" and key != "prompt":
                result[f"prev_{key}"] = entry[key]

    return result

# Function to generate text using the Qwen model (LLaVA-OneVision)
def generate_text_by_vlm(
    video_path,
    model,
    tokenizer,
    image_processor,
    question,
    max_frames_num=16,
    conv_template="qwen_1_5",
    max_num=5,
    if_add_complete_list=False,
    event_list_len=None,
    seed=None,
):
    # Function to get the prompt by appending the question to the conversation template
    def process_prompt(question, conv_template="qwen_1_5"):
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()
    

    start = time.time()
    video_frames, _, _ = load_video(
        video_path,
        max_frames_num,
        1,
        force_sample=True,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )
    print(
        f"#### Local VLM: Processing {video_path}, extracted frames shape: {video_frames.shape}"
    )
    image_tensors = []
    frames = image_processor.preprocess(video_frames, return_tensors="pt")["pixel_values"].half()

    image_tensors.append(frames)
    image_sizes = [Image.fromarray(frame).size for frame in video_frames]

    prompt_question = process_prompt(question, conv_template)

    input_ids = tokenizer_image_token(
        prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0)

    # random seed
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # Generate response with retry logic
    test_time = 0
    while test_time < max_num:
        with torch.no_grad():
            cont = model.generate(
                input_ids,
                images=image_tensors,
                image_sizes=image_sizes,
                do_sample=True,
                # temperature=0,
                temperature=1.0, # to allow some randomness
                max_new_tokens=4096,
                modalities=["video"],
            )

        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
        end = time.time()
        print(f"Time taken for {video_path}: {end - start:.2f} seconds")

        generated_text = text_outputs[0]

        if not if_add_complete_list and len(generated_text) > 0:
            # If not checking the completion list, return the generated text
            return generated_text
        else:
            # Extract the completion list
            print(f"Scoring output:\n{generated_text}\n")
            complete_list = get_complete_list_from_output(generated_text)

            if event_list_len is not None and len(complete_list) == event_list_len:
                return generated_text
            else:
                print(f"-> ERROR: Completion list length {len(complete_list)} does not match event list length {event_list_len}, retrying...")

        test_time += 1

    else:
        print(f"-- ERROR: Completion list length does not match after {max_num} retries, skipping...")
        return generated_text


# use Katna to extract keyframes
def generate_result_by_gpt4o(
    api_args,
    model_dir,
    video_path,
    question,
    max_frames_num=8,
    max_num=3,
    if_add_complete_list=False,
    event_list_len=None,
    seed=None,
):

    images_base64 = load_video_by_opencv2(
        model_dir,
        video_path,
        max_frames_num,
        target_width=TARGET_WIDTH,
        target_height=TARGET_HEIGHT,
    )

    if len(images_base64) == 0:
        if not if_add_complete_list:
            return ""
        else:
            return [], ""

    # Prepare the API request
    api_base = api_args["api_base"]  # Replace with your Azure endpoint
    deployment_name = api_args["deployment_name"]  # Replace with your deployment name
    api_version = "2024-03-01-preview"  # Replace with the correct API version
    constructed_url = f"{api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={api_version}"
    headers = {
        "Content-Type": "application/json",
        "api-key": api_args["api_key"],  # Replace with your actual API key
    }

    def run_api(inputs):
        request = requests.post(constructed_url, headers=headers, json=inputs)
        response = request.json()
        return response

    def get_inputs(question, images_base64, seed=None):
        body = [
            {
                "role": "system",
                "content": "You are an expert in assisting humans. Follow the user prompt in a completion mode. Generate precise and clear response. End your response with {END}.",
            },
            {"role": "user", "content": question},
            {"role": "user", "content": images_base64},
        ]

        inputs = {}
        inputs["messages"] = body  # For "chat"
        inputs["max_tokens"] = 2000
        inputs["stop"] = "{END}"
        if seed is not None:
            inputs["user"] = f"seed_{seed}"

        return inputs
    
    inputs = get_inputs(question, images_base64, seed=seed)


    test_time = 0
    reason = None
    while test_time < max_num:
        results = run_api(inputs)
        try:
            generated_text = results["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error in API response: {e}")
            print(f"API response: {results}")
            generated_text = ""
            reason = results

        if not if_add_complete_list and len(generated_text) > 0:
            # If not checking the completion list, return the generated text
            return generated_text
        else:
            # Extract the completion list
            print(f"Scoring output:\n{generated_text}\n")
            complete_list = get_complete_list_from_output(generated_text)

            if event_list_len is not None and len(complete_list) == event_list_len:
                return complete_list, generated_text
            else:
                print(f"!!! ERROR: Completion list length {len(complete_list)} does not match event list length {event_list_len}, retrying...")

        test_time += 1

    else:
        print(f"= ERROR: Completion list length does not match after {max_num} retries, skipping...")
        final_reason = reason if reason is not None else "Completion list length does not match"
        print(f"Error: {final_reason}")

        if not if_add_complete_list:
            return f"Error: {final_reason}"
        else:
            return [], f"Error: {final_reason}"


def get_complete_list_from_output(scoring_output):
    # Step 4: Extract completion list from scoring_output and calculate completion score
    # Find all occurrences of [COMPLETE_LIST]: ...
    complete_list_matches = re.findall(r"\[COMPLETE_LIST\]:\s*(.*)", scoring_output)
    if complete_list_matches:
        # Take the last occurrence
        complete_list_str = complete_list_matches[-1].strip()
        # Convert to list of integers
        completion_list = [int(x.strip()) for x in re.findall(r"\d+", complete_list_str)]
    else:
        completion_list = []
        print(f"Could not extract completion list from scoring output")

    return completion_list


def process_videos_in_directory(
    api_args,
    base_dir,
    model_dir,
    prompts_dict,
    output_file,
    model,
    tokenizer,
    image_processor,
    first_template,
    general_template,
    max_frames_num=16,
    conv_template="qwen_1_5",
    eval_model_type="llava",
    if_reeval=0,
    if_backup=1,
    if_reeval_null=0,
    max_num=3, # max number of retries
    repeat_time=3, # repeat time for each video
):
    # Only process files listed in prompts_dict
    model_full_path = os.path.join(base_dir, model_dir)
    results = {}
    prompts_dict = dict(sorted(prompts_dict.items()))

    

    # Load existing results if the file exists
    if os.path.exists(output_file) and if_reeval == 0:
        with open(output_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        # Backup the original output file
        if if_backup == 1:
            shutil.copy(output_file, output_file.replace(".json", "_old.json"))
            print(f"Backup completed!")

        # Remove outputs with null completion list if required
        if if_reeval_null == 1:
            original_len = len(results)
            for video_file in list(results.keys()):
                for idx in range(repeat_time):
                    output_key = f"output{idx}"
                    if output_key in results[video_file]:
                        if results[video_file][output_key]["completion_list"] == []:
                            del results[video_file][output_key]
                # If all outputs are removed, remove the video entry
                if not any(f"output{idx}" in results[video_file] for idx in range(repeat_time)):
                    del results[video_file]
            print(f"Remove {original_len - len(results)} entries with null completion list, original length: {original_len}, new length: {len(results)}")


    for video_file, entry in prompts_dict.items():
        prompt = entry["prompt"]
        event_list = entry["event_list"]
        video_path = os.path.join(model_full_path, video_file)

        # Initialize results[video_file] if not present
        if video_file not in results:
            results[video_file] = {
                "prompt": prompt,
                "event_list": event_list,
            }
            results[video_file] = add_other_keys_in_entry(results[video_file], entry)
            # init the completion_list_avg and completion_score_avg
            results[video_file]["completion_list_avg"] = []
            results[video_file]["completion_score_avg"] = None

        # For collecting outputs to compute averages
        completion_list_repeat = []
        completion_score_repeat = []

        # Now process repeats
        for idx in range(repeat_time):
            output_key = f"output{idx}"

            # Skip the outputs that have been processed
            if output_key in results[video_file] and if_reeval == 0:
                print(f"Skip {video_file} {output_key} since it has been processed.")
                # Collect the existing completion_list and completion_score for averaging
                completion_list = results[video_file][output_key]["completion_list"]
                completion_score = results[video_file][output_key]["completion_score"]
                if completion_list != [] and completion_score is not None:
                    completion_list_repeat.append(completion_list)
                    completion_score_repeat.append(completion_score)
                continue

            print(f"Processing {video_file} {output_key}...")
            seed = idx  # Use idx as seed for different repeats



            try:
                if not os.path.exists(video_path):
                    print(f"Video file {video_path} does not exist. Skipping.")
                    reason = f"Error: no such video"

                    # Store error output
                    results[video_file][output_key] = {
                        "completion_list": [],
                        "completion_score": None,
                        "description": "",
                        "scoring_output": reason,
                        "seed": seed,
                    }
                    continue

                # Step 1: Generate the description of the video
                description_question = f"{DEFAULT_IMAGE_TOKEN}\n{first_template}"

                if eval_model_type == "llava":
                    description = generate_text_by_vlm(
                        video_path,
                        model,
                        tokenizer,
                        image_processor,
                        description_question,
                        max_frames_num=max_frames_num,
                        conv_template=conv_template,
                        if_add_complete_list=False,
                        seed=seed,
                    )
                elif eval_model_type == "gpt4o":
                    description = generate_result_by_gpt4o(
                        api_args,
                        model_dir,
                        video_path,
                        description_question,
                        max_frames_num=max_frames_num,
                        if_add_complete_list=False,
                        seed=seed,
                    )
                else:
                    raise ValueError("Unsupported model type")

                print(f"Description for {video_file}:\n{description}\n")

                # Step 2: Combine description with the prompt and event list into the general_template
                events_formatted = "\n".join(
                    [f"{i+1}. {event}" for i, event in enumerate(event_list)]
                )
                scoring_question = f"{DEFAULT_IMAGE_TOKEN}\n" + general_template.format(
                    description.strip(), prompt, len(event_list), events_formatted
                )
                print(f"Prompt: {prompt}, Events: {event_list}")

                # Step 3: Generate the scoring output
                if eval_model_type == "llava":
                    scoring_output = generate_text_by_vlm(
                        video_path,
                        model,
                        tokenizer,
                        image_processor,
                        scoring_question,
                        max_frames_num=max_frames_num,
                        conv_template=conv_template,
                        max_num=max_num,
                        if_add_complete_list=True,
                        event_list_len=len(event_list),
                        seed=seed,
                    )
                    completion_list = get_complete_list_from_output(scoring_output)

                elif eval_model_type == "gpt4o":
                    # Pass the seed to the function
                    scoring_output = ""
                    completion_list, scoring_output = generate_result_by_gpt4o(
                        api_args,
                        model_dir,
                        video_path,
                        scoring_question,
                        max_frames_num=max_frames_num,
                        if_add_complete_list=True,
                        event_list_len=len(event_list),
                        max_num=max_num,
                        seed=seed,
                    )
                    if completion_list == []:
                        reason = scoring_output
                else:
                    raise ValueError("Unsupported model type")

                # Calculate completion score
                completion_score = (
                    sum(completion_list) / len(completion_list)
                    if len(completion_list) > 0
                    else None
                )

                print(f"Completion list for {video_file} {output_key}: {completion_list}")
                print(f"Completion score for {video_file} {output_key}: {completion_score}")

                if completion_list != [] and completion_score is not None:
                    completion_list_repeat.append(completion_list)
                    completion_score_repeat.append(completion_score)

                # Store the output
                results[video_file][output_key] = {
                    "completion_list": completion_list,
                    "completion_score": completion_score,
                    "description": description.strip(),
                    "scoring_output": scoring_output,
                    "seed": seed,
                }

            except Exception as e:
                print(f"Error processing {video_file} {output_key}: {e}")
                reason = f"Error: {e}"

                # Store error output
                results[video_file][output_key] = {
                    "completion_list": [],
                    "completion_score": None,
                    "description": "",
                    "scoring_output": reason,
                    "seed": seed,
                }
                continue

            # Update averages after each successful repeat
            if completion_list_repeat:
                # Compute the element-wise average of the completion lists
                completion_list_avg = [
                    sum(values) / len(values) 
                    for values in zip(*completion_list_repeat)
                ]
                completion_score_avg = sum(completion_score_repeat) / len(completion_score_repeat)
            else:
                completion_list_avg = []
                completion_score_avg = None

            results[video_file]["completion_list_avg"] = completion_list_avg
            results[video_file]["completion_score_avg"] = completion_score_avg

            print(f'------------------ final average completion list: {completion_list_avg}, completion score: {completion_score_avg} ------------------')

            # Write the results dictionary to the output file after each output
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

        # At the end of processing each video_file, write the results
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

    # At the end of the function, write the results again
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    return results


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="Evaluate video completion scores using LLaVA-NeXT or GPT4o model"
    )
    # Basic arguments
    parser.add_argument(
        "--videos_dir",
        type=str,
        help="Directory containing the generated videos",
        default="./generated_videos",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="Directory containing the evaluation results",
        default="./results",
    )
    parser.add_argument(
        "--prompts_file_name",
        type=str,
        help="Prompts file",
    )
    parser.add_argument(
        "--eval_model_type",
        type=str,
        default="llava",
        choices=["llava", "gpt4o"],
        help="Model used for evaluation, GPT4o or LLaVA-OV-Chat-72B",
    )

    # LLaVA arguments
    parser.add_argument(
        "--pretrained",
        type=str,
        default="lmms-lab/llava-onevision-qwen2-72b-ov-chat",
        help="Pretrained model name",
    )
    parser.add_argument(
        "--llava_model_name", 
        type=str, 
        default="llava_qwen", 
        help="Model name"
    )
    parser.add_argument(
        "--device_map", 
        type=str, 
        default="auto", 
        help="Device map for the model"
    )

    # GPT4o arguments
    parser.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key for GPT4o model",
    )
    parser.add_argument(
        "--api_base",
        type=str,
        default="",
        help="API base URL for GPT4o model",
    )
    parser.add_argument(
        "--deployment_name",
        type=str,
        default="",
        help="Deployment name for GPT4o model",
    )

    # Eval Details
    parser.add_argument(
        "--max_frames_num",
        type=int,
        default=32,
        help="Maximum number of frames to process per video",
    )
    parser.add_argument(
        "--generative_model_list",
        type=str,
        default="kling::videocraft2",
        help="Models to process, separated by '::' (e.g., 'kling::videocraft2')",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default="",
        help="Additional postfix to add to the output file name",
    )

    # Evaluation options
    parser.add_argument(
        "--if_reeval",
        type=int,
        default=0,
        help="If re-evaluate the results or not if the evaluation file already exists",
    )
    parser.add_argument(
        "--if_backup",
        type=int,
        default=1,
        help="If backup the original output file (with _old.json postfix)",
    )
    parser.add_argument(
        "--if_reeval_null",
        type=int,
        default=1,
        help="If re-evaluate the videos that have null completion list",
    )
    parser.add_argument(
        "--repeat_time",
        type=int,
        default=3,
        help="Repeat time for each video",
    )

    # Summarization options
    parser.add_argument(
        "--vote_type",
        type=int,
        default=1,
        help="Vote type for summarization, see details in summarization.py. 0 -> no voting; 1 -> a event is considered completed iff all evaluation try think the event is completed",
    )
    parser.add_argument(
        "--null_type",
        type=int,
        default=0,
        help="Null type for summarization, see details in summarization.py. 0 -> consider both empty list and non-empty list; 1 -> only consider non-empty list",
    )


    args = parser.parse_args()

    # Read the prompts from 'basic_prompts.json'
    prompts_file = os.path.join('./prompts', args.prompts_file_name)
    with open(prompts_file, "r", encoding="utf-8") as f:
        prompts_dict = json.load(f)

    # Prepare conversation input

    first_template = """
    Please describe the given key frames in the video in detail, in temporal order. The video may be generated by some video generative model rather than sampling from the real world, so it may be vague or not clear. You can point out if you don't see the video clearly.
    """

    general_template = """
    This is the description of the video you generated before, please refer to it to complete the following tasks.
    {}

    Now, based on these descriptions and the video, you are asked to accurately determine if the following generated video fulfills the requirements of the prompt. The prompt contains several (2~4) events, you need to judge if each event is strictly completed in the video. If the event is completed, please mark it as 1, otherwise, mark it as 0.
    For example, if the prompt is: "A man dribbles a basketball and then throws it in a court", the prompt describes two events: "A man dribbles a basketball" and "And then the man throws the basketball in a court". But if the video generated using this prompt only accomplishes dribbling or only accomplishes shooting, then the completion list is [1, 0]. If you think both events are not completed, the completion list is [0, 0], etc.

    Please judge whether the event are completed very strictly. If you think an item is blurry, hard to identify, or the action is vague, you should judge it as not completed. And please explain the reasons in detail before you give out the score.
    You also need to check the item consistency between different events. If the prompt implies that the subject (or object) in different events should be the same, but in the video they are different, you should mark the later event in the prompt as not completed. For example, for the above prompt, if the man that dribbles the basketball is different from the man that throws the basketball (should be the same people, but video shows two different people), or the basketball that's dribbled is different from the basketball that's thrown (should be the same object, but video shows two different objects), you must mark the later event 'throwing the ball' as not completed.
    
    Remember, you should judge whether the events are completed very strictly. And you should first provide the reasons or analysis for each event, and then give out the list of completion flag for each event (0 or 1 for uncompleted or completed).
    Please remember to output the complete list at the end of output again, strictly follow the format: 'Finally we have [COMPLETE_LIST]: 1, 0' in a single line.

    Now, let's begin scoring! The prompt is '{}', there are {} events:
    {}
    """

    # Get list of subdirectories (models)
    generative_model_list = args.generative_model_list.split("::")
    subdirectories = [
        d
        for d in generative_model_list
        if os.path.isdir(os.path.join(args.videos_dir, d))
    ]
    print(f"Subdirectories: {subdirectories}")

    # make sure the results_dir exists
    os.makedirs(args.results_dir, exist_ok=True)

    if args.eval_model_type == 'llava':
        from LLaVA_NeXT.llava.model.builder import load_pretrained_model
        from LLaVA_NeXT.llava.mm_utils import tokenizer_image_token
        from LLaVA_NeXT.llava.constants import IMAGE_TOKEN_INDEX
        from LLaVA_NeXT.llava.conversation import conv_templates

        api_args = None

    elif args.eval_model_type == 'gpt4o':
        api_args = {
            "api_key": args.api_key,
            "api_base": args.api_base,
            "deployment_name": args.deployment_name,
        }

    print(f"================= Start Processing =================")

    # Initialize variables
    model = None
    tokenizer = None
    image_processor = None

    if args.eval_model_type == "llava":
        # Load the Qwen model (LLaVA-OneVision)
        tokenizer, model, image_processor, max_length = load_pretrained_model(
            args.pretrained,
            None,
            args.llava_model_name,
            device_map=args.device_map,
            attn_implementation="sdpa",
        )
        model.eval()
    elif args.eval_model_type == "gpt4o": # For GPT4o, no need to load a model locally
        pass
    else:
        raise ValueError("Unsupported model type")

    avg_score_list = []
    null_completion_list = []
    null_name_list = []
    results_list = []

    for model_dir in subdirectories:
        # The output file is named after the model (sub-directory) and saved in the results_dir
        output_file = get_output_file_path(
            args.results_dir, model_dir, args.eval_model_type, args.output_postfix
        )

        print(f"================= Processing model directory: {model_dir} =================")
        # Process all videos in the model directory
        results = process_videos_in_directory(
            api_args,
            args.videos_dir,
            model_dir,
            prompts_dict,
            output_file,
            model,
            tokenizer,
            image_processor,
            first_template,
            general_template,
            max_frames_num=args.max_frames_num,
            conv_template="qwen_1_5",
            eval_model_type=args.eval_model_type,
            if_reeval=args.if_reeval,
            if_backup=args.if_backup,
            if_reeval_null=args.if_reeval_null,
            repeat_time=args.repeat_time,
        )

    # print(f"================= Final Summarization =================")
    for model_dir in subdirectories:
        final_summarize(
            args.results_dir,
            eval_model_type = args.eval_model_type, 
            output_postfix = args.output_postfix, 
            generative_model_name = model_dir,   
            vote_type = args.vote_type,
            null_type = args.null_type,
        )


