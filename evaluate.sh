#### debug mode for gpt4o ####
python evaluate.py \
    --eval_model_type gpt4o \
    --api_key "" \
    --api_base "" \
    --deployment_name "" \
    --generative_model_list pika1.5::hailuo \
    --prompts_file_name prompts_debug.json \
    --output_postfix '_debug' \
    --repeat_time 1 # how many independent evaluation runs for each video 

# #### debug mode for llava ####
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python evaluate.py \
#     --eval_model_type llava \
#     --generative_model_list pika1.5::hailuo \
#     --prompts_file_name prompts_debug.json \
#     --output_postfix '_debug' \
#     --repeat_time 2


# #### full mode for gpt4o ####
# python evaluate.py \
#     --eval_model_type gpt4o \
#     --api_key "" \
#     --api_base "" \
#     --deployment_name "" \
#     --generative_model_list X \
#     --prompts_file_name all_prompts.json \
#     --output_postfix '_final' \
#     --repeat_time 3 # standard setting in StoryEval paper

# #### full mode for llava ####
# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python evaluate.py \
#     --eval_model_type llava \
#     --generative_model_list X \
#     --prompts_file_name all_prompts.json \
#     --output_postfix '_final' \
#     --repeat_time 2 # standard setting in StoryEval paper
