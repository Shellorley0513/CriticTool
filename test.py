import os
import json
import argparse
from tqdm import tqdm
from critictool.utils.prompt import *
from critictool.llms.openAI import GPT
from critictool.llms.huggingface import HFTransformer
from critictool.evaluators import LLMEvaluator
from critictool.evaluators import EnvironmentEvaluator
from critictool.utils.format import format_load


def parse_args():
    parser = argparse.ArgumentParser()
    # hf means huggingface, if you want to use huggingface model, you should specify the path of the model
    parser.add_argument('--model_type', type=str, choices=['api', 'hf'], default='hf')
    # [api model name: 'gpt-3.5-turbo-16k', 'gpt-4o-2024-08-06', 'claude-3-5-sonnet-20241022']
    parser.add_argument('--model_path', type=str, help='path to huggingface model / api model name')
    parser.add_argument('--api_key', type=str, default='sk-xxxxx')
    parser.add_argument('--base_url', type=str, default='https://api.openai.com/v1/chat/completions')
    parser.add_argument('--eval', type=str, choices=['environment', 'llm'])
    parser.add_argument('--cot', default=False, action='store_true')
    parser.add_argument('--dataset_path', type=str, default='.CriticTool-Dataset/internal_error/base_data_wo_cot.json')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--out_name', type=str, default='tmp.json')
    parser.add_argument('--out_dir', type=str, default="work_dirs/")
    parser.add_argument('--test_num', type=int, default=-1, help='number of samples to test, -1 means all')
    parser.add_argument('--template', type=str, default='api')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--environment_mode', type=str, choices=['skip', 'finish'])
    parser.add_argument('--retry_limit', type=int, default=3, help='number of retries allowed in case of environment errors')
    args = parser.parse_args()
    return args


def load_dataset(dataset_path, out_dir, is_resume=False, tmp_folder_name='tmp'):
    with open(dataset_path, 'r', encoding='utf-8') as file:
        dataset = json.load(file)
    total_num = len(dataset.items())
    tested_num = 0
    if is_resume:
        file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
        for filename in file_list:
            if filename.split('.')[0] in dataset:
                tested_num += 1
                file_id = filename.split('.')[0]
                dataset.pop(file_id)
            else:
                print(f"Warning: {filename} not in dataset, remove it from cache")
                os.remove(os.path.join(out_dir, tmp_folder_name, filename))
    return dataset, tested_num, total_num



def insert_system_prompt(prompt, mode="llm", cot=False):
    if mode == "skip":
        if cot:
            new_content = ENV_SKIP_COT_PROMPT
        else:
            new_content = ENV_SKIP_PROMPT
    elif mode == "finish":
        if cot:
            new_content = ENV_FINISH_COT_PROMPT
        else:
            new_content = ENV_FINISH_PROMPT
    elif mode == "llm":
        if cot:
            new_content = LLM_COT_PROMPT
        else:
            new_content = LLM_PROMPT
    insert_idx = prompt[0]["content"].find("Remember: only generate ONE step each time")
    prompt[0]["content"] =  prompt[0]["content"][:insert_idx] + new_content + prompt[0]["content"][insert_idx:]


def environment_infer(dataset, llm, out_dir, cot, retry_limit=3, tmp_folder_name='tmp', test_num = 1, mode='skip', batch_size=1):
    test_list = list(dataset.keys())[:test_num]
    batch_infer_list = []; batch_infer_ids = []
    for idx in tqdm(test_list):
        prompt = dataset[idx]['query_prompt']
        insert_system_prompt(prompt, mode, cot)
        dataset[idx]['predictions'] = []
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        if len(batch_infer_ids) == batch_size or idx == test_list[-1]:
            infer_time = 0
            while(infer_time <= retry_limit):
                predictions_list = llm.chat(batch_infer_list)
                infer_time += 1
                del_list = []
                for ptr, prediction in enumerate(predictions_list):
                    data_ptr = batch_infer_ids[ptr]
                    format_pred = format_load(prediction)
                    if isinstance(format_pred, dict) and 'name' in format_pred.keys() and 'args' in format_pred.keys():
                        if format_pred['name'] == dataset[data_ptr]['ground_truth'][0]['name'] and format_pred['args'] == dataset[data_ptr]['ground_truth'][0]['args']:
                            batch_infer_list[ptr].append({"role": "assistant",
                                                            "content": str(prediction)})
                            batch_infer_list[ptr].append(batch_infer_list[ptr][-2])
                        else:
                            del_list.append(ptr)
                    else:
                        del_list.append(ptr)
                    dataset[data_ptr]['predictions'].append(prediction)
                    with open(os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'), 'w', encoding='utf-8') as file:
                        json.dump(dataset[data_ptr], file, ensure_ascii=False, indent=2)
                for del_ptr in sorted(del_list, reverse=True):
                    del batch_infer_list[del_ptr]
                    del batch_infer_ids[del_ptr]
                if batch_infer_list == []:
                    break
            batch_infer_list = []; batch_infer_ids = []
    # load results from cache
    results = dict()
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        with open(os.path.join(out_dir, tmp_folder_name, filename), 'r', encoding='utf-8') as file:
            results[file_id] = json.load(file)
    return results


def llm_infer(dataset, llm, out_dir, cot, tmp_folder_name='tmp', test_num = 1, batch_size=1):
    test_list = list(dataset.keys())[:test_num]
    batch_infer_list = []; batch_infer_ids = []
    for idx in tqdm(test_list):
        prompt = dataset[idx]['query_prompt']
        insert_system_prompt(prompt, 'llm', cot)
        batch_infer_list.append(prompt)
        batch_infer_ids.append(idx)
        if len(batch_infer_ids) == batch_size or idx == test_list[-1]:
            predictions = llm.chat(batch_infer_list)
            for ptr, prediction in enumerate(predictions):
                data_ptr = batch_infer_ids[ptr]
                dataset[data_ptr]['predictions'] = prediction
                with open(os.path.join(out_dir, tmp_folder_name, f'{data_ptr}.json'), 'w', encoding='utf-8') as file:
                    json.dump(dataset[data_ptr], file, ensure_ascii=False, indent=2)
            batch_infer_ids = []; batch_infer_list = []
        # load results from cache
    results = dict()
    file_list = os.listdir(os.path.join(out_dir, tmp_folder_name))
    for filename in file_list:
        file_id = filename.split('.')[0]
        with open(os.path.join(out_dir, tmp_folder_name, filename), 'r', encoding='utf-8') as file:
            results[file_id] = json.load(file)
    return results
        


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tmp_folder_name = os.path.splitext(args.out_name)[0]
    os.makedirs(os.path.join(args.out_dir, tmp_folder_name), exist_ok=True)
    dataset, tested_num, total_num = load_dataset(args.dataset_path, args.out_dir, args.resume, tmp_folder_name=tmp_folder_name)
    if args.test_num == -1:
        test_num = max(total_num - tested_num, 0)
    else:
        test_num = max(min(args.test_num - tested_num, total_num - tested_num), 0)
    print(f"Tested {tested_num} samples, left {test_num} samples, total {total_num} samples")
    
    output_file_path = os.path.join(args.out_dir, args.out_name)
    if test_num != 0:
        if args.model_type == 'hf':
            llm = HFTransformer(model_path=args.model_path, template=args.template)
        elif args.model_type == 'api':
            llm = GPT(model=args.model_path, key=args.api_key, openai_api_base=args.base_url)
        if args.eval == 'environment':
            prediction = environment_infer(dataset, llm, args.out_dir, retry_limit=args.retry_limit, cot=args.cot, tmp_folder_name=tmp_folder_name, test_num=test_num, mode=args.environment_mode, batch_size=args.batch_size)
        elif args.eval == 'llm':
            prediction = llm_infer(dataset, llm, args.out_dir, cot=args.cot, tmp_folder_name=tmp_folder_name, test_num=test_num, batch_size=args.batch_size)
        with open(output_file_path, 'w', encoding='utf-8') as file:
            json.dump(prediction, file, ensure_ascii=False, indent=2)
    else:
        if not os.path.exists(output_file_path):
            raise FileNotFoundError(
                f"Output file not found or corrupted: {output_file_path}. "
                "Please disable `--is_resume` and re-run to regenerate results."
            )

    eval_result_path = "results.jsonl"
    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, args.out_name.split('.json')[0] + '_eval' + '.json')
    if args.eval == "llm":
        evaluator = LLMEvaluator(output_file_path, cot=args.cot)
    elif args.eval == "environment":
        evaluator = EnvironmentEvaluator(output_file_path, mode=args.environment_mode, cot=args.cot)
    eval_results, mean_eval_results = evaluator.evaluate()
    print(mean_eval_results)
    print(f"Writing Mean Evaluation Results to {json_path} and {eval_result_path}")
    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(eval_results, file, ensure_ascii=False, indent=2)
    
    record = {
        "config": vars(args),
        "metrics": mean_eval_results,
    }
    with open(eval_result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
