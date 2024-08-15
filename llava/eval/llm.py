import os
import abc
import asyncio
import math
import tiktoken
import openai
import backoff
from abc import abstractmethod
from openai import OpenAIError, RateLimitError, APIError
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
            'api_key': os.getenv("OPENAI_API_KEY"),
        },
    }

    deployment_max_length_dict = {
        'gpt-4': 8192,
        'gpt-4-32k': 32768,
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16384,
    }

    def __init__(self, model_id):
        self.temperature = 0.0
        self.top_k = 1
        self.encoding = tiktoken.encoding_for_model(model_id)
        self.openai_api = 'default'
        self.model_id = model_id
        self.max_length = self.deployment_max_length_dict[model_id]
        self.client = openai.AsyncOpenAI(api_key=self.openai_cxn_dict[self.openai_api]['api_key'])

    def gen_messages(self, fixed_instruction, few_shot_examples, input, input_header, output_header):
        messages = [
            {
                "role": "system",
                "content": fixed_instruction,
            },
        ]
        for example in few_shot_examples:
            messages.extend(
                [
                    {
                        "role": "user",
                        "content": input_header + '\n' + example['user'] + '\n\n' + output_header,
                    },
                    {
                        "role": "assistant",
                        "content": example['assistant'],
                    },
                ]
            )
        messages.extend(
            [
                {
                    "role": "user",
                    "content": input_header + '\n' + input + '\n\n' + output_header,
                },
            ]
        )
        return messages

    @backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_tries=5, on_backoff=lambda details: logger.info(f"Backing off {details['wait']} seconds after {details['tries']} tries."))
    async def make_api_call_to_gpt(self, messages):
        try:
            response = await self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error making API call: {e}")
            raise

    async def dispatch_openai_requests(self, messages_list):
        tasks = [self.make_api_call_to_gpt(messages) for messages in messages_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

    def infer(self, messages_list):
        return asyncio.run(self.dispatch_openai_requests(messages_list))

    def split_input(self, fixed_instruction, few_shot_examples, splittable_input, input_header, output_header):
        fixed_token_ids = self.encoding.encode(fixed_instruction + ' '.join([x['user'] + ' ' + x['assistant'] for x in few_shot_examples]))
        remaining_token_len = math.ceil((self.prompt_percent * self.max_length) - len(fixed_token_ids))
        split_token_ids = self.encoding.encode(splittable_input)
        split_token_ids_list = [split_token_ids[i:i + remaining_token_len + 10] for i in range(0, len(split_token_ids), remaining_token_len)]
        split_input_list = [self.encoding.decode(split_token_ids) for split_token_ids in split_token_ids_list]
        return [self.gen_messages(fixed_instruction, few_shot_examples, split_input, input_header, output_header) for split_input in split_input_list]
