import json
import ast

def format_load(raw_data: str, start_character: str = '', end_character: str = ''):
    if "```json" in raw_data:
        raw_data = raw_data[raw_data.find("```json") + len("```json"):]
        raw_data = raw_data.strip("`").strip()
    if start_character != '':
        raw_data = raw_data[raw_data.find(start_character):]
    if end_character != '':
        raw_data = raw_data[:raw_data.rfind(end_character) + len(end_character)]
    successful_parse = False
    try:
        data = ast.literal_eval(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(raw_data.replace("\'", "\""))
        successful_parse = True
    except Exception as e:
        pass
    if not successful_parse:
        pass
    prepared_json_format = dict()
    try:
        prepared_json_format['error'] = data['error']
    except:
        prepared_json_format['error'] = ''
    try:
        prepared_json_format['thought'] = str(data['thought'])
    except Exception as e:
        prepared_json_format['thought'] = ''
    try:
        prepared_json_format['name'] = str(data['name'])
    except Exception as e:
        prepared_json_format['name'] = ''

    try:
        prepared_json_format['args'] = data['args']
    except:
        prepared_json_format['args'] = ''
    return prepared_json_format
    
def get_function_list(system_prompt: str):
    start = system_prompt[:system_prompt.find("[")]
    end = system_prompt[system_prompt.find("\nTo use a tool"):]
    tool_config = system_prompt[len(start):-len(end)]
    try:
        tool_config = json.loads(tool_config)
    except:
        tool_config = ast.literal_eval(tool_config)
    return tool_config