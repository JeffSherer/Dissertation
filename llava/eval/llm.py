import abc
import asyncio
import math
import tiktoken
import openai
import backoff
from abc import abstractmethod
from openai import OpenAIError, RateLimitError, APIError

class LLM(abc.ABC):
    prompt_percent = 0.9

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def infer(self, prompts):
        raise NotImplementedError("Subclasses should implement this!")

    @abstractmethod
    def split_input(self, fixed_instruction, few_shot_examples, splittable_input, input_header, output_header):
        raise NotImplementedError("Subclasses should implement this!")

class GPT(LLM):
    prompt_percent = 0.8

    openai_cxn_dict = {
        'default': {
            'endpoint': "INSERT YOUR AZURE OPENAI ENDPOINT HERE",
            'api_key': "INSERT YOUR AZURE OPENAI API KEY HERE",
        },
    }

    deployment_max_length_dict = {
        'gpt-4': 8192,
        'gpt-4-0314': 8192,
        'gpt-4-32k': 32768,
        'gpt-35-turbo': 4096,
        'gpt-35-turbo-16k': 16385,
    }

    def __init__(self, model_id):
        self.temperature = 0.0
        self.top_k = 1
        self.encoding = tiktoken.encoding_for_model("-".join(model_id.split("-", 2)[:2]).replace('5', '.5'))
        self.openai_api = 'default'
        self.model_id = model_id
        self.max_length = self.deployment_max_length_dict[model_id]
        self.client = openai.AsyncAzureOpenAI(
            api_key=self.openai_cxn_dict[self.openai_api]['api_key'],
            api_version="2023-12-01-preview",
            azure_endpoint=self.openai_cxn_dict[self.openai_api]['endpoint']
        )

    def gen_messages(self, fixed_instruction, few_shot_examples, input, input_header, output_header):
        messages = [
            {"role": "system", "content": fixed_instruction},
        ]
        for example in few_shot_examples:
            messages.extend([
                {"role": "user", "content": f'{input_header}\n{example["user"]}\n\n{output_header}'},
                {"role": "assistant", "content": example["assistant"]}
            ])
        messages.append({"role": "user", "content": f'{input_header}\n{input}\n\n{output_header}'})
        return messages

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5)
    async def make_api_call_to_gpt(self, messages):
        response = await self.client.completions.create(
            model=self.model_id,
            prompt=messages,
            temperature=self.temperature
        )
        return response.choices[0].message.content

    async def dispatch_openai_requests(self, messages_list):
        tasks = [self.make_api_call_to_gpt(messages) for messages in messages_list]
        results = await asyncio.gather(*tasks)
        return results

    def infer(self, messages_list):
        return asyncio.run(self.dispatch_openai_requests(messages_list))

    def split_input(self, fixed_instruction, few_shot_examples, splittable_input, input_header, output_header):
        fixed_token_ids = self.encoding.encode(fixed_instruction + ' '.join([x['user'] + ' ' + x['assistant'] for x in few_shot_examples]))
        remaining_token_len = math.ceil((self.prompt_percent * self.max_length) - len(fixed_token_ids))

        split_token_ids = self.encoding.encode(splittable_input)
        split_token_ids_list = [split_token_ids[i:i + remaining_token_len + 10] for i in range(0, len(split_token_ids), remaining_token_len)]
        split_input_list = [self.encoding.decode(ids) for ids in split_token_ids_list]

        return [self.gen_messages(fixed_instruction, few_shot_examples, split_input, input_header, output_header) for split_input in split_input_list]
