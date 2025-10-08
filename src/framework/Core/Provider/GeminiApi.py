from __future__ import annotations

import asyncio
from typing import Dict, List, Optional, Tuple

import google.generativeai as genai

from Config.LLMConfig import LLMConfig, LLMType
from Core.Common.Constants import USE_CONFIG_TIMEOUT
from Core.Common.Logger import log_llm_stream, logger
from Core.Provider.BaseLLM import BaseLLM
from Core.Provider.LLMProviderRegister import register_provider


def _chunk_text(chunk) -> str:
    """Extract textual content from a Gemini chunk or response part."""

    if getattr(chunk, "text", None):
        return chunk.text

    texts: List[str] = []
    parts = getattr(chunk, "parts", None)
    if parts:
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)

    candidates = getattr(chunk, "candidates", None)
    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content and getattr(content, "parts", None):
                for part in content.parts:
                    text = getattr(part, "text", None)
                    if text:
                        texts.append(text)

    return "".join(texts)


def _usage_from_response(response) -> Optional[Dict[str, int]]:
    usage = getattr(response, "usage_metadata", None)
    if not usage:
        return None

    return {
        "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }


def _combine_generation_config(config: LLMConfig, max_tokens: Optional[int]) -> Dict:
    generation_config: Dict[str, float | int] = {
        "temperature": config.temperature,
        "top_p": config.top_p,
    }

    if config.top_k:
        generation_config["top_k"] = config.top_k

    max_output_tokens = max_tokens or config.max_token
    if max_output_tokens:
        generation_config["max_output_tokens"] = max_output_tokens

    return generation_config


def _prepare_prompt(messages: List[Dict[str, str]]) -> Tuple[Optional[str], List[Dict]]:
    system_messages: List[str] = []
    contents: List[Dict] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            system_messages.append(content)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": content}]})

    system_instruction = "\n\n".join(system_messages) if system_messages else None
    return system_instruction, contents


@register_provider(LLMType.GEMINI)
class GeminiLLM(BaseLLM):
    """Google Gemini provider compatible with the project LLM interface."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.model = config.model or "gemini-1.5-pro"
        self.cost_manager = None
        self.semaphore = asyncio.Semaphore(config.max_concurrent)

        client_options = None
        if config.base_url:
            client_options = {"api_endpoint": config.base_url}

        genai.configure(api_key=config.api_key, client_options=client_options)

    async def _achat_completion(
        self,
        messages: List[Dict],
        timeout: int = USE_CONFIG_TIMEOUT,
        max_tokens: Optional[int] = None,
        format: str = "text",
    ) -> Dict:
        text, usage = await self._generate_response(messages, stream=False, timeout=timeout, max_tokens=max_tokens)

        response: Dict = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": text,
                    }
                }
            ]
        }

        if usage:
            response["usage"] = usage
            self._update_costs(usage, model=self.model, local_calc_usage=False)

        return response

    async def _achat_completion_stream(
        self,
        messages: List[Dict],
        timeout: int = USE_CONFIG_TIMEOUT,
        max_tokens: Optional[int] = None,
        format: str = "text",
    ) -> str:
        text, usage = await self._generate_response(messages, stream=True, timeout=timeout, max_tokens=max_tokens)

        if usage:
            self._update_costs(usage, model=self.model, local_calc_usage=False)

        return text

    async def acompletion(self, messages: List[Dict], timeout: int = USE_CONFIG_TIMEOUT) -> Dict:
        return await self._achat_completion(messages, timeout=timeout)

    def get_choice_text(self, rsp: Dict) -> str:
        return rsp.get("choices", [{}])[0].get("message", {}).get("content", "")

    def get_choice_delta_text(self, rsp: Dict) -> str:
        return rsp.get("choices", [{}])[0].get("delta", {}).get("content", "")

    async def _generate_response(
        self,
        messages: List[Dict],
        stream: bool,
        timeout: int,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        system_instruction, contents = _prepare_prompt(messages)
        generation_config = _combine_generation_config(self.config, max_tokens)

        if not contents:
            contents = [{"role": "user", "parts": [{"text": ""}]}]

        def _run_generation() -> Tuple[str, Optional[Dict[str, int]]]:
            model_kwargs = {
                "model_name": self.model,
                "generation_config": generation_config,
            }
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(**model_kwargs)

            if stream:
                try:
                    response_stream = model.generate_content(contents, stream=True)
                except Exception as exc:  # pragma: no cover - defensive logging
                    logger.exception("Gemini streaming generation failed: %s", exc)
                    raise
                full_text_parts: List[str] = []
                usage_details: Optional[Dict[str, int]] = None
                for chunk in response_stream:
                    chunk_text = _chunk_text(chunk)
                    if chunk_text:
                        log_llm_stream(chunk_text)
                        full_text_parts.append(chunk_text)
                    chunk_usage = _usage_from_response(chunk)
                    if chunk_usage:
                        usage_details = chunk_usage

                log_llm_stream("\n")
                return "".join(full_text_parts), usage_details

            try:
                response = model.generate_content(contents)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Gemini generation failed: %s", exc)
                raise

            text = _chunk_text(response) or getattr(response, "text", "")
            usage_details = _usage_from_response(response)
            return text, usage_details

        timeout_seconds = self.get_timeout(timeout)
        text, usage = await asyncio.wait_for(asyncio.to_thread(_run_generation), timeout_seconds)

        return text, usage
