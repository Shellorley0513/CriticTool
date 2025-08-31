#!/bin/sh

# Add the '--cot' parameter to enable CoT evaluation if the dataset_path matches '*_w_cot.json'

# hf model
model_path=Qwen/Qwen2.5-7B-Instruct
out_dir=work_dirs/Qwen2.5-7B-Instruct
template=qwen #llama2 for agentlm and toolLLaMA, llama3 for ToolACE
batchsize=10

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval llm \
--dataset_path ./CriticTool-Dataset/internal_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name interal_base_data_wo_cot.json \
--test_num -1 \
--template $template \
--batch_size $batchsize

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval llm \
--dataset_path ./CriticTool-Dataset/internal_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name interal_evol_data_wo_cot.json \
--test_num -1 \
--template $template \
--batch_size $batchsize

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_skip_base_data_wo_cot.json \
--test_num -1 \
--template $template \
--environment_mode skip \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_finish_base_data_wo_cot.json \
--test_num -1 \
--template $template \
--environment_mode finish \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_skip_evol_data_wo_cot.json \
--test_num -1 \
--template $template \
--environment_mode skip \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "hf" \
--model_path $model_path \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_finish_evol_data_wo_cot.json \
--test_num -1 \
--template $template \
--environment_mode finish \
--batch_size $BATCHSIZE \
--retry_limit 3


# api model
model_path=gpt-4o
api_key=sk-xxxxx
base_url=https://api.openai.com/v1/chat/completions
out_dir=work_dirs/gpt-4o
batchsize=10

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval llm \
--dataset_path ./CriticTool-Dataset/internal_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name interal_base_data_wo_cot.json \
--test_num -1 \
--batch_size $batchsize

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval llm \
--dataset_path ./CriticTool-Dataset/internal_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name interal_evol_data_wo_cot.json \
--test_num -1 \
--batch_size $batchsize

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_skip_base_data_wo_cot.json \
--test_num -1 \
--environment_mode skip \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/base_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_finish_base_data_wo_cot.json \
--test_num -1 \
--environment_mode finish \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_skip_evol_data_wo_cot.json \
--test_num -1 \
--environment_mode skip \
--batch_size $BATCHSIZE \
--retry_limit 3

python test.py \
--model_type "api" \
--model_path $model_path \
--api_key $api_key \
--base_url $base_url \
--eval environment \
--dataset_path ./CriticTool-Dataset/external_error/evol_data_wo_cot.json \
--resume \
--out_dir $out_dir \
--out_name external_finish_evol_data_wo_cot.json \
--test_num -1 \
--environment_mode finish \
--batch_size $BATCHSIZE \
--retry_limit 3