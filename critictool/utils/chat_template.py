from typing import List, Dict

class Meta_Template:
    meta_template_dict = dict(
        api = [
            dict(role='system', api_role='system'),
            dict(role='user', api_role='user'),
            dict(role='function', api_role='user')
        ],
        llama2 = [
            dict(role='system', begin='<s>[INST] <<SYS>>\n', end='\n<</SYS>>\n\n'),
            dict(role='user', begin='<s>[INST] ', end=' [/INST]'),
            dict(role='function', begin='<s>[INST] ', end=' [/INST]'),
            dict(role='assistant', begin=' ', end=' </s>\n'),
        ],
        llama3 = [
            dict(role='system', begin='<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n', end='<|eot_id|>'),
            dict(role='user', begin='<|start_header_id|>user<|end_header_id|>\n\n', end='<|eot_id|>'),
            dict(role='function', begin='<|start_header_id|>assistant<|end_header_id|>\n\n', end='<|eot_id|>'),
            dict(role='assistant', begin='<|start_header_id|>assistant<|end_header_id|>\n\n', end='<|eot_id|>'),
        ],
        qwen = [
            dict(role='system', begin='<|im_start|>system\n', end='<|im_end|>'),
            dict(role='user', begin='\n<|im_start|>user\n', end='<|im_end|>'),
            dict(role='function', begin='\n<|im_start|>user\n', end='<|im_end|>'),
            dict(role='assistant', begin='\n<|im_start|>assistant\n', end='<|im_end|>'),
        ],
        glm4 = [
            dict(role='system', begin='[gMASK]<sop><|system|>\n', end=''),
            dict(role='user', begin='<|user|>\n', end=''),
            dict(role='function', begin='<|user|>\n', end=''),
            dict(role='assistant', begin='<|assistant|>\n', end=''),
        ],
        ministral = [
            dict(role='system', begin='', end='\n\n'),
            dict(role='user', begin='[INST]', end='[/INST]'),
            dict(role='function', begin='[INST]', end='[/INST]'),
            dict(role='assistant', begin='', end=' </s>'),
        ]
    )
    def __init__(
            self,
            template_type: str = 'api'
        ) -> None:
        self.template_type = template_type
        if template_type == 'api':
            self.role_map = {item["role"]: item["api_role"] for item in self.meta_template_dict[template_type]}
        else:
            self.role_map = {item["role"]: {"begin": item["begin"], "end": item["end"]} for item in self.meta_template_dict[template_type]}

    def apply_api_role(self, messages: List[Dict]) -> List[Dict]:
        new_messages = []
        for msg in messages:
            new_msg = msg.copy()
            new_msg["role"] = self.role_map[new_msg["role"]]
            new_messages.append(new_msg)
        return messages
    
    def apply_hf_template(self, messages: List[Dict]) -> str:
        wrapped_text = ""
        for msg in messages:
            role_info = self.role_map[msg["role"]]
            if self.template_type == "llama2" and wrapped_text.endswith('\n<</SYS>>\n\n'):
                wrapped_text += f"{msg['content']}{role_info['end']}"
            elif self.template_type == "ministral" and msg["role"] == "system":
                wrapped_text += "<s>"
            else:
                wrapped_text += f"{role_info['begin']}{msg['content']}{role_info['end']}"
        if self.template_type == "ministral":
            pos = wrapped_text.rfind(self.role_map["user"]["begin"])+len(self.role_map["user"]["begin"])
            system_prompt = self.role_map["system"]["end"] + messages[0]["content"] + self.role_map["system"]["end"]
            wrapped_text = wrapped_text[:pos] + system_prompt + wrapped_text[pos:]
        wrapped_text += f"{self.role_map['assistant']['begin']}"
        return wrapped_text