import ast
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from critictool.utils.format import format_load, get_function_list

class LLMEvaluator:
    """LLM Evaluator
    Args:
        dataset_path(str): File path of evaluation dataset
        bert_score_model(str): the bert_score model for sentence similarity, default = "all-mpnet-base-v2". 
            Refer to https://www.sbert.net/docs/pretrained_models.html for more models.
    """
    def __init__(
        self,
        dataset_path: str,
        cot: bool = False,
        bert_score_model: str = "all-mpnet-base-v2",
    ) -> None:
        self.cot = cot
        self.dataset_path = dataset_path
        self.bert_score_model = bert_score_model
        self.sentence_model = SentenceTransformer(self.bert_score_model)

        self.error_num = 0

    def _load_dataset(self):
        self.dataset = []
        with open(self.dataset_path, 'r', encoding='utf-8') as file:
            dataset = json.load(file)
        for key in dataset.keys():
            datum = dataset[key]
            self.dataset.append(self._process_response(datum, key))
    

    def _process_response(self, datum, key):
        error = datum["error_type"]
        gt_data = datum["ground_truth"]
        pred_data = format_load(datum["predictions"], start_character='{', end_character='}')
        function_list = get_function_list(datum['query_prompt'][0]['content'])
        return dict(query=key, function_list=function_list, gt_data=gt_data, pred_data=pred_data, error=error)
    
    def _get_function_list(self, system_prompt: str):
        start = system_prompt[:system_prompt.find("[")]
        end = system_prompt[system_prompt.find("\nTo use a tool"):]
        tool_config = system_prompt[len(start):-len(end)]
        try:
            tool_config = json.loads(tool_config)
        except:
            tool_config = ast.literal_eval(tool_config)
        return tool_config

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
            metric_keys = ["critic", "reflect", "name", "args", "goal", "parse_rate"]
        else:
            metric_keys = ["critic", "reflect", "name", "args", "parse_rate"]
        
        metrics_results = dict()
        for data in tqdm(results_list):
            metrics_results[data['query']] = {metric_keys[x]: 0 for x in range(len(metric_keys))}
            if self.cot:
                if data['pred_data']['thought'] and data['pred_data']['name'] and \
                    (isinstance(data['pred_data']['args'], dict) or data['pred_data']['args']):
                    metrics_results[data['query']]['parse_rate'] = 1
            else:
                if data['pred_data']['name'] and \
                    (isinstance(data['pred_data']['args'], dict) or data['pred_data']['args']):
                    metrics_results[data['query']]['parse_rate'] = 1
            if not data['error']:
                if not data['pred_data']['error'] and \
                    metrics_results[data['query']]['parse_rate']:
                    metrics_results[data['query']]['critic'] = 1
            else:
                self.error_num += 1
                if data['pred_data']['error']:
                    metrics_results[data['query']]['critic'] = 1
                    if data['pred_data']['error'] in ['tool_select_error', 'tool_hallucination_error', 'parameters_value_error', 'parameters_key_error']:
                        if data['error'] == data['pred_data']['error']:
                            metrics_results[data['query']]['reflect'] = 1
                if self.cot:
                    if data['pred_data']['thought']:
                        metrics_results[data['query']]['goal'] = self._compute_similarity(data['pred_data']['thought'], data['gt_data']['thought'])
                if data['pred_data']['name']:
                    if data['pred_data']['name'] == data['gt_data']['name']:
                        metrics_results[data['query']]['name'] = 1
                        metrics_results[data['query']]['args'] = self._arg_eval(data['function_list'], data['pred_data'], data['gt_data'])

        results = dict()
        for key in metric_keys:
            results[key] = 0
            for query in metrics_results.keys():
                results[key] += metrics_results[query][key]
            if key in ['critic', 'parse_rate']:
                results[key] /= len(metrics_results)
            else:
                results[key] /= self.error_num
            results[key] = float(results[key])
        return metrics_results, results


    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            results_list.append(data_sample)
        return self._post_process(results_list)