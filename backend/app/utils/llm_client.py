"""
LLM客户端封装
统一使用OpenAI格式调用
"""

import json
import re
import time
import logging
from typing import Optional, Dict, Any, List
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError

from ..config import Config

logger = logging.getLogger('mirofish.llm_client')


class LLMClient:
    """LLM客户端"""

    # 可重试的 HTTP 状态码（服务端过载 / 限流）
    _RETRYABLE_STATUS_CODES = {429, 529}
    # 最大重试次数（不含首次请求）
    _MAX_RETRIES = 10
    # 重试基础等待秒数，指数退避: base * 2^attempt
    _RETRY_BASE_WAIT = 5

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model = model or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY 未配置")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _should_retry(self, exc: Exception) -> bool:
        """判断异常是否可重试（429 限流 / 529 过载 / 网络错误）"""
        if isinstance(exc, APIStatusError):
            return exc.status_code in self._RETRYABLE_STATUS_CODES
        if isinstance(exc, (APITimeoutError, APIConnectionError)):
            return True
        return False

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: Optional[Dict] = None
    ) -> str:
        """
        发送聊天请求（自动重试 429/529 错误）

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            response_format: 响应格式（如JSON模式）

        Returns:
            模型响应文本
        """
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_format:
            kwargs["response_format"] = response_format

        last_exc = None
        for attempt in range(self._MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                # 部分模型会在content中包含思考内容，需要移除
                content = re.sub(r'<think_>[\s\S]*?<_/think_>', '', content).strip()
                return content
            except (APIStatusError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                if self._should_retry(e) and attempt < self._MAX_RETRIES:
                    wait = self._RETRY_BASE_WAIT * (2 ** attempt)
                    logger.warning(
                        f"LLM API {e.status_code} 错误，{wait}s 后重试 "
                        f"({attempt + 1}/{self._MAX_RETRIES})"
                    )
                    time.sleep(wait)
                else:
                    raise
        raise last_exc

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096
    ) -> Dict[str, Any]:
        """
        发送聊天请求并返回JSON

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            解析后的JSON对象
        """
        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        # 清理markdown代码块标记
        cleaned_response = response.strip()
        cleaned_response = re.sub(r'^```(?:json)?\s*\n?', '', cleaned_response, flags=re.IGNORECASE)
        cleaned_response = re.sub(r'\n?```\s*$', '', cleaned_response)
        cleaned_response = cleaned_response.strip()

        try:
            return json.loads(cleaned_response)
        except json.JSONDecodeError:
            raise ValueError(f"LLM返回的JSON格式无效: {cleaned_response}")
