import json


class LocalPostConditionsModel:
    def __init__(self, model_path):
      
        self.model_path = model_path
        self.model = None
        self.test_data = {}

        if not model_path or model_path == 'test':
            if model_path == 'test':
                with open('oracle_tests.json', 'r') as f:
                    self.test_data = json.load(f)
        elif model_path.startswith(('openai', 'neulab', 'litellm')):
            self.model = model_path
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.bfloat16, device_map='auto'
            )

    async def generate_postconditions(self, prompt, trajectory=None):
        # When in test mode, prompt is treated as instance_id
        if self.model_path == 'test':
            entry = self.test_data.get(prompt, {})
            test_file = entry.get('test_file', 'No test file available').strip()
            return (
                f'TESTS:\n{test_file}\n'
            )
