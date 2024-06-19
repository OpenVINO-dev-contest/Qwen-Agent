import os
from http import HTTPStatus
from pprint import pformat
from typing import Dict, Iterator, List, Optional
from threading import Thread

from optimum.intel.openvino import OVModelForCausalLM
from transformers import (AutoTokenizer, AutoConfig,
                          TextIteratorStreamer, StoppingCriteriaList, StoppingCriteria)

from qwen_agent.llm.base import ModelServiceError, register_llm
from qwen_agent.llm.schema import ASSISTANT, DEFAULT_SYSTEM_MESSAGE, SYSTEM, USER, Message
from qwen_agent.llm.text_base import BaseTextChatModel
from qwen_agent.log import logger


from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria


class StopSequenceCriteria(StoppingCriteria):
    """
    This class can be used to stop generation whenever a sequence of tokens is encountered.

    Args:
        stop_sequences (`str` or `List[str]`):
            The sequence (or list of sequences) on which to stop execution.
        tokenizer:
            The tokenizer used to decode the model outputs.
    """

    def __init__(self, stop_sequences, tokenizer):
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        self.stop_sequences = stop_sequences
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        decoded_output = self.tokenizer.decode(input_ids.tolist()[0])
        return any(decoded_output.endswith(stop_sequence) for stop_sequence in self.stop_sequences)

@register_llm('openvino')
class OpenVINO(BaseTextChatModel):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.ov_model = OVModelForCausalLM.from_pretrained(
            cfg.get('model_dir', 'model'),
            device=cfg.get('device', 'cpu'),
            ov_config=cfg.get('ov_config', {}),
            config=AutoConfig.from_pretrained(cfg.get('model_dir', 'model')),
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["model_dir"])

    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool,
        generate_cfg: dict,
    ) -> Iterator[List[Message]]:
        
        prompt = self._build_text_completion_prompt(messages)
        logger.debug(f'*{prompt}*')
        input_token = self.tokenizer(prompt, return_tensors="pt").input_ids
        generate_cfg["input_ids"] = input_token
        generate_cfg['stopping_criteria'] = generate_cfg['stop']
        streamer = TextIteratorStreamer(
            self.tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)
        generate_cfg["streamer"] = streamer

        def generate_and_signal_complete():
            self.ov_model.generate(**generate_cfg)
        t1 = Thread(target=generate_and_signal_complete)
        t1.start()

        if delta_stream:
            last_len = 0
            delay_len = 5
            in_delay = False
            text = ''
            for new_text in streamer:
                if (len(new_text) - last_len) <= delay_len:
                    in_delay = True
                    continue
                else:
                    in_delay = False
                    real_text = new_text[:-delay_len]
                    now_rsp = real_text[last_len:]
                    yield [Message(ASSISTANT, now_rsp)]
                    last_len = len(real_text)
            if text and (in_delay or (last_len != len(new_text))):
                yield [Message(ASSISTANT, new_text[last_len:])]

        else:
            for new_text in streamer:
                yield [Message(ASSISTANT, new_text)]

    def _chat_no_stream(
        self,
        messages: List[Message],
        generate_cfg: dict,
    ) -> List[Message]:
        prompt = self._build_text_completion_prompt(messages)
        logger.debug(f'*{prompt}*')
        input_token = self.tokenizer(prompt, return_tensors="pt").input_ids
        generate_cfg["input_ids"] = input_token
        generate_cfg['stopping_criteria'] = generate_cfg['stop']
        response = self.ov_model.generate(**generate_cfg)
        return [Message(ASSISTANT, response)]

    @staticmethod
    def _build_text_completion_prompt(messages: List[Message]) -> str:
        im_start = '<|im_start|>'
        im_end = '<|im_end|>'
        if messages[0].role == SYSTEM:
            sys = messages[0].content
            assert isinstance(sys, str)
            prompt = f'{im_start}{SYSTEM}\n{sys}{im_end}'
        else:
            prompt = f'{im_start}{SYSTEM}\n{DEFAULT_SYSTEM_MESSAGE}{im_end}'
        if messages[-1].role != ASSISTANT:
            messages.append(Message(ASSISTANT, ''))
        for msg in messages:
            assert isinstance(msg.content, str)
            if msg.role == USER:
                query = msg.content.lstrip('\n').rstrip()
                prompt += f'\n{im_start}{USER}\n{query}{im_end}'
            elif msg.role == ASSISTANT:
                response = msg.content.lstrip('\n').rstrip()
                prompt += f'\n{im_start}{ASSISTANT}\n{response}{im_end}'
        assert prompt.endswith(im_end)
        prompt = prompt[:-len(im_end)]
        return prompt
