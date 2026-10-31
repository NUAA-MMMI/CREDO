"""
Temp-R1 Search Agent Loop.
Custom AgentLoop for verl that implements multi-turn retrieval-augmented generation
using <search>query</search> and <answer>answer</answer> tags.
"""

import asyncio
import logging
import os
import re
from typing import Any, Optional
from uuid import uuid4

import requests
import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.utils.profiler import simple_timer
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("search_agent")
class SearchAgentLoop(AgentLoopBase):
    """
    Multi-turn search agent loop for Temp-R1.
    
    The model generates text that may contain:
    - <search>query</search> → triggers retrieval, results appended as <information>...</information>
    - <answer>answer</answer> → terminates the loop
    
    This replaces the old LLMGenerationManager from search_r1.
    """

    def __init__(self, *args, 
                 search_url: str = "RETRIEVAL_ENDPOINT",
                 topk: int = 30,
                 max_turns: int = 3,
                 max_obs_length: int = 1200,
                 max_start_length: int = 2000,
                 prompt_length: int = 5200,
                 response_length: int = 3000,
                 max_tokens_per_turn: int = 1000,
                 max_model_len: int = 10000,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # Some VERL agent-loop releases do not forward custom Hydra constructor
        # fields consistently. CF-PSR owns an explicit endpoint variable rather
        # than inheriting other experimental runtime state.
        self.search_url = os.environ.get("CREDO_SEARCH_URL", search_url)
        self.require_retrieval = os.environ.get("CREDO_REQUIRE_RETRIEVAL", "1") == "1"
        self.topk = topk
        self.max_turns = max_turns
        self.max_obs_length = max_obs_length
        self.max_start_length = max_start_length
        
        self.prompt_length = prompt_length
        self.response_length = response_length
        self.max_tokens_per_turn = max_tokens_per_turn
        self.max_model_len = max_model_len

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Main agent loop: generate → search → generate → ... → answer."""
        messages = list(kwargs["raw_prompt"])
        metrics = {}
        request_id = uuid4().hex

        # Initialize state
        prompt_ids = await self.apply_chat_template(messages)
        # Match legacy Temp-R1 Stage1: the dataset may admit prompts up to
        # max_prompt_length=5200, while the actual agent loop starts from the
        # rightmost max_start_length=2000 tokens.
        prompt_ids = prompt_ids[-self.max_start_length:]
        # The returned prompt must remain the immutable initial policy prefix.
        # Deriving it by subtracting a later truncated response can leak an
        # overflow suffix into the prompt after a long retrieval observation.
        initial_prompt_ids = list(prompt_ids)
        response_ids = []
        response_mask = []
        response_logprobs = []
        
        num_turns = 0
        num_searches = 0
        
        for turn in range(self.max_turns + 1):  # +1 for final answer turn
            remaining_response_tokens = self.response_length - len(response_ids)
            remaining_context_tokens = self.max_model_len - len(prompt_ids)
            max_turn_tokens = min(
                remaining_response_tokens,
                remaining_context_tokens,
                self.max_tokens_per_turn,
            )
            if max_turn_tokens <= 0:
                break

            turn_sampling_params = dict(sampling_params)
            turn_sampling_params["max_tokens"] = max_turn_tokens
            turn_sampling_params["stop"] = ["</search>", "</answer>"]
            turn_sampling_params["include_stop_str_in_output"] = True

            # Generate
            with simple_timer("generate_sequences", metrics):
                output: TokenOutput = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=prompt_ids,
                    sampling_params=turn_sampling_params,
                )
            
            num_turns += 1
            new_ids = output.token_ids
            
            # Decode to check for search/answer tags
            new_text = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.tokenizer.decode(new_ids, skip_special_tokens=True)
            )
            
            # Post-process: truncate at first </search> or </answer>
            processed_text = self._postprocess_response(new_text)
            
            # Re-tokenize the processed text
            processed_ids = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.tokenizer.encode(processed_text, add_special_tokens=False)
            )
            
            # Update state
            prompt_ids += processed_ids
            response_ids += processed_ids
            response_mask += [1] * len(processed_ids)
            if output.log_probs:
                # Only take logprobs for the tokens we kept
                response_logprobs += output.log_probs[:len(processed_ids)]
            
            # Check termination
            if len(response_mask) >= self.response_length:
                break
            
            # Parse action
            action, content = self._parse_action(processed_text)

            # Match official Temp-R1: the final model call is generation-only.
            if turn == self.max_turns:
                if action == 'search':
                    num_searches += 1
                break
            
            if action == 'answer':
                # Done
                break
            elif action == 'search':
                # Execute search
                num_searches += 1
                with simple_timer("tool_calls", metrics):
                    search_result = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: self._do_search(content)
                    )
                
                # Format observation
                obs_text = f'\n\n<information>{search_result.strip()}</information>\n\n'
                obs_ids = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.tokenizer.encode(obs_text, add_special_tokens=False)
                )
                
                # Truncate observation if too long
                if len(obs_ids) > self.max_obs_length:
                    obs_ids = obs_ids[:self.max_obs_length]
                
                # Add observation to prompt (masked - not part of model's output)
                prompt_ids += obs_ids
                response_ids += obs_ids
                response_mask += [0] * len(obs_ids)  # mask out retrieval results
                if response_logprobs:
                    response_logprobs += [0.0] * len(obs_ids)
                
                if len(response_mask) >= self.response_length:
                    break
            else:
                # Invalid action - add feedback
                feedback = ('\nMy previous action is invalid. '
                          'If I want to search, I should put the query between <search> and </search>. '
                          'If I want to give the final answer, I should put the answer between <answer> and </answer>. '
                          'Let me try again.\n')
                feedback_ids = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.tokenizer.encode(feedback, add_special_tokens=False)
                )
                prompt_ids += feedback_ids
                response_ids += feedback_ids
                response_mask += [0] * len(feedback_ids)
                if response_logprobs:
                    response_logprobs += [0.0] * len(feedback_ids)
                
                if len(response_mask) >= self.response_length:
                    break

        # Truncate to response_length
        response_ids = response_ids[:self.response_length]
        response_mask = response_mask[:self.response_length]
        if response_logprobs:
            response_logprobs = response_logprobs[:self.response_length]
        
        prompt_only_ids = initial_prompt_ids

        assert len(response_ids) <= self.response_length
        assert len(response_mask) == len(response_ids)
        assert len(prompt_only_ids) <= self.prompt_length
        
        metrics["num_turns"] = num_turns
        metrics["num_searches"] = num_searches
        
        extra_fields = {}
        if output and hasattr(output, 'extra_fields'):
            extra_fields.update(output.extra_fields)
        
        return AgentLoopOutput(
            prompt_ids=prompt_only_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data={},
            mm_processor_kwargs={},
            response_logprobs=response_logprobs if response_logprobs else None,
            num_turns=num_turns,
            metrics=metrics,
            extra_fields=extra_fields,
        )

    def _postprocess_response(self, text: str) -> str:
        """Truncate response at first </search> or </answer> tag."""
        if '</search>' in text:
            return text.split('</search>')[0] + '</search>'
        elif '</answer>' in text:
            return text.split('</answer>')[0] + '</answer>'
        return text

    def _parse_action(self, text: str):
        """Parse <search>...</search> or <answer>...</answer> from text."""
        pattern = r'<(search|answer)>(.*?)</\1>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            action = match.group(1)
            content = match.group(2).strip()
            return action, content
        return None, None

    def _do_search(self, query: str) -> str:
        """Call the retrieval service."""
        try:
            payload = {
                "queries": [query],
                "topk": self.topk,
                "return_scores": True
            }
            response = requests.post(self.search_url, json=payload, timeout=30)
            results = response.json()['result'][0]
            return self._format_results(results)
        except Exception as e:
            logger.warning(f"Search failed: {e}")
            if self.require_retrieval:
                raise RuntimeError(
                    f"Temp-R1 OCOR required retriever is unavailable at {self.search_url}: {e}"
                ) from e
            return f"Search error: {e}"

    def _format_results(self, retrieval_result) -> str:
        """Format retrieval results into readable text."""
        format_reference = ''
        for idx, doc_item in enumerate(retrieval_result):
            content = doc_item['document']['contents']
            title = content.split("\n")[0]
            text = "\n".join(content.split("\n")[1:])
            format_reference += f"Doc {idx+1}(Title: {title}) {text}\n"
        return format_reference
