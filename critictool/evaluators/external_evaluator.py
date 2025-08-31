import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from critictool.utils.format import format_load, get_function_list


class EnvironmentEvaluator:
    """Environment Evaluator
    Args:
        dataset_path(str): File path of evaluation dataset
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2". 
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        mode: str = 'skip',
        cot: bool = False,
        bert_score_model: str = "all-mpnet-base-v2",
    ) -> None:
        self.cot = cot
        self.mode = mode
        self.dataset_path = dataset_path
        self.sentence_model = SentenceTransformer(bert_score_model)


    def _load_dataset(self):
        self.dataset = []
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        for key in dataset.keys():
            datum = dataset[key]
            self.dataset.append(self._process_response(datum, key))
    

    def _process_response(self, datum, key):
        pred_data = []
        gt_data = []
        for gt in datum["ground_truth"]:
            gt_data.append(gt)
        for pred in datum["predictions"]:
            pred_data.append(format_load(pred, start_character='{', end_character='}'))
        function_list = get_function_list(datum['query_prompt'][0]['content'])
        return dict(query=key, function_list=function_list, gt_data=gt_data, pred_data=pred_data)

    def _compute_similarity(self, pred_val, gt_val):
        arg_content = [str(pred_val), str(gt_val)]
        arg_emb = self.sentence_model.encode(arg_content, convert_to_tensor=True)
        cosine_score = np.maximum(util.cos_sim(arg_emb[0], arg_emb[1]).cpu().numpy(), 0)
        return float(cosine_score[0, 0])
    
    def _arg_eval(self, function_list, pred_data, gt_data):
        def contains_chinese(text: str) -> bool:
            return any('\u4e00' <= char <= '\u9fff' for char in text)
        arg_score = 0
        args_type = dict()
        required_args = []
        optional_args = []
        TYPE_MAP = {
            'NUMBER': int, 'integer': int,
            'FLOAT': float, 'float': float,
            'BOOLEAN': bool, 'boolean': bool,
            'STRING': str, 'string': str,
            'array': list,
            'dict': dict,
            'any': 'any'
        }
        func = next((f for f in function_list if f['name'] == pred_data['name']), None)
        if not func:
            return 0.0
        for param_type in ['required_parameters', 'optional_parameters']:
            param_list = func.get(param_type, [])
            for arg in param_list:
                if param_type == 'required_parameters':
                    required_args.append(arg['name'])
                else:
                    optional_args.append(arg['name'])
                args_type[arg['name']] = TYPE_MAP.get(arg['type'], str)
        if not isinstance(pred_data['args'], dict):
            return 0.0
        if len(gt_data['args']) == 0:
            arg_score = 1
            for arg in pred_data['args']:
                if arg not in required_args and arg not in optional_args:
                    arg_score = 0
        else: 
            for arg in gt_data['args']:
                if arg in pred_data['args']:
                    if args_type[arg] == 'any' or \
                        isinstance(pred_data['args'][arg], args_type[arg]):
                        if str(pred_data['args'][arg]) == str(gt_data['args'][arg]):
                            arg_score += 1
                        else:
                            arg_score += self._compute_similarity(pred_data['args'][arg], gt_data['args'][arg])
                    elif type(pred_data['args'][arg]) == type(gt_data['args'][arg]) \
                        and isinstance(gt_data['args'][arg], str):
                        if not contains_chinese(str(gt_data['args'][arg])) and not contains_chinese(str(pred_data['args'][arg])): 
                            arg_score += self._compute_similarity(pred_data['args'][arg], gt_data['args'][arg])
                else:
                    if arg not in required_args:
                        arg_score += 0.5
            arg_score = arg_score/len(gt_data['args'])
        return arg_score

    def _post_process(self, results_list):
        if self.cot:
            if self.mode == 'finish':
                metric_keys = ['retry', 'break', 'thought', 'name', 'parse_rate']
            elif self.mode == 'skip':
                metric_keys = ['retry', 'break', 'thought', 'name', 'args', 'parse_rate']
        else:
            if self.mode == 'finish':
                metric_keys = ['retry', 'break', 'name', 'parse_rate']
            elif self.mode == 'skip':
                metric_keys = ['retry', 'break', 'name', 'args', 'parse_rate']
    
        metrics_results = dict()
        for data in tqdm(results_list):
            metrics_results[data['query']] = {metric_keys[x]: 0 for x in range(len(metric_keys))}
            metrics_results[data['query']]['parse_rate'] = 1
            for item in data['pred_data']:
                if self.cot:
                    if not item['thought'] or not item['name'] or (not isinstance(item['args'], dict) and not item['args']):
                        metrics_results[data['query']]['parse_rate'] = 0
                else:
                    if not item['name'] or (not isinstance(item['args'], dict) and not item['args']):
                        metrics_results[data['query']]['parse_rate'] = 0

            for response in data['pred_data']:
                if response['name'] == data['gt_data'][0]['name'] and response['args'] == data['gt_data'][0]['args']:
                    metrics_results[data['query']]['retry'] = 1
                else:
                    if metrics_results[data['query']]['parse_rate'] == 1:
                        metrics_results[data['query']]['break'] = 1
                    if self.mode == 'skip':
                        if self.cot:
                            metrics_results[data['query']]['thought'] = self._compute_similarity(
                                response['thought'], data['gt_data'][1]['thought']
                            )
                        if response['name'] == data['gt_data'][1]['name']:
                            metrics_results[data['query']]['name'] = 1
                            if response['name'] != 'FinishAction':
                                metrics_results[data['query']]['args'] = self._arg_eval(data['function_list'], response, data['gt_data'][1])
                            else:
                                metrics_results[data['query']]['args'] = self._compute_similarity(
                                    response['args'], data['gt_data'][1]['args']
                                )
                    elif self.mode == 'finish':
                        critic_thought = f"I encountered an issue while trying to call{data['gt_data'][0]['name']}, please let me know how you would like to proceed."
                        if self.cot:
                            metrics_results[data['query']]['thought'] = self._compute_similarity(
                                str(response['thought']), critic_thought
                            )
                        if response['name'] == 'FinishAction':
                            metrics_results[data['query']]['name'] = 1
                            if self.cot:
                                arg_cosine_score = self._compute_similarity(
                                    str(response['args']), critic_thought
                                )
                                if arg_cosine_score > metrics_results[data['query']]['thought']:
                                    metrics_results[data['query']]['thought'] = arg_cosine_score
        results = dict()
        for key in metric_keys:
            results[key] = 0
            for query in metrics_results.keys():
                results[key] += metrics_results[query][key]
            results[key] /= len(metrics_results)
        return metrics_results, results


    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            results_list.append(data_sample)
        return self._post_process(results_list)