"""Pydantic models for OpenAI-compatible Chat Completions API."""
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class MessageContent(BaseModel):
    """A text content part in a message."""
    type: Literal["text"] = "text"
    text: str


class ImageURL(BaseModel):
    """Image URL content detail."""
    url: str
    detail: Optional[Literal["auto", "low", "high"]] = "auto"


class ImageContentPart(BaseModel):
    """An image content part in a message."""
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


class InputAudioContent(BaseModel):
    """Input audio content in a message."""
    data: str  # Base64 encoded
    format: Literal["wav", "mp3"]


class InputAudioPart(BaseModel):
    """An input audio content part in a message."""
    type: Literal["input_audio"] = "input_audio"
    input_audio: InputAudioContent


class FileContentPart(BaseModel):
    """A file content part in a message."""
    type: Literal["file"] = "file"
    file: Dict[str, Any]  # Contains file_data, file_id, or filename


class RefusalContentPart(BaseModel):
    """A refusal content part in an assistant message."""
    type: Literal["refusal"] = "refusal"
    refusal: str


class FunctionCall(BaseModel):
    """A function call within a message."""
    name: str
    arguments: str  # JSON formatted


class ToolCallFunction(BaseModel):
    """A function tool call."""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ToolCallCustom(BaseModel):
    """A custom tool call."""
    id: str
    type: Literal["custom"] = "custom"
    custom: Dict[str, Any]  # Contains name and input


class ToolCall(BaseModel):
    """A tool call (function or custom)."""
    id: str
    type: str
    function: Optional[FunctionCall] = None
    custom: Optional[Dict[str, Any]] = None


class AudioContent(BaseModel):
    """Audio content in a message."""
    id: str
    data: Optional[str] = None
    expires_at: Optional[float] = None
    transcript: Optional[str] = None


class FunctionTool(BaseModel):
    """A function tool definition."""
    type: Literal["function"] = "function"
    function: Dict[str, Any]  # Contains name, description, parameters, strict


class CustomTool(BaseModel):
    """A custom tool definition."""
    type: Literal["custom"] = "custom"
    custom: Dict[str, Any]  # Contains name, description, format


class ChatCompletionTool(BaseModel):
    """A tool definition for chat completions."""
    type: Literal["function", "custom"]
    function: Optional[Dict[str, Any]] = None
    custom: Optional[Dict[str, Any]] = None


class StreamOptions(BaseModel):
    """Stream options for chat completions."""
    include_usage: Optional[bool] = None
    include_obfuscation: Optional[bool] = None


class ResponseFormat(BaseModel):
    """Response format specification."""
    type: Literal["json_object", "json_schema"]
    json_schema: Optional[Dict[str, Any]] = None


class PredictionContent(BaseModel):
    """Prediction content specification."""
    type: Literal["content"] = "content"
    content: Union[str, List[Dict[str, Any]]]


class WebSearchOptions(BaseModel):
    """Web search options."""
    search_context_size: Optional[Literal["low", "medium", "high"]] = None
    user_location: Optional[Dict[str, Any]] = None


class CreateChatCompletionRequest(BaseModel):
    """Request body for chat completions."""
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    n: Optional[int] = Field(None, ge=1, le=128)
    stream: Optional[bool] = False
    stream_options: Optional[StreamOptions] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(None, ge=-2.0, le=2.0)
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = Field(None, ge=0, le=20)
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ChatCompletionTool]] = None
    tool_choice: Optional[Union[Literal["none", "auto", "required"], Dict[str, Any]]] = None
    response_format: Optional[ResponseFormat] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    audio: Optional[Dict[str, Any]] = None
    reasoning_effort: Optional[Literal["none", "minimal", "low", "medium", "high", "xhigh"]] = None
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None
    store: Optional[bool] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    safety_identifier: Optional[str] = None
    prompt_cache_key: Optional[str] = None
    prompt_cache_retention: Optional[Literal["in_memory", "24h"]] = None
    seed: Optional[float] = None
    prediction: Optional[PredictionContent] = None
    parallel_tool_calls: Optional[bool] = None
    web_search_options: Optional[WebSearchOptions] = None
    verbosity: Optional[Literal["low", "medium", "high"]] = None
    logit_bias: Optional[Dict[int, float]] = None


class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: Optional[Dict[str, Any]] = None
    prompt_tokens_details: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    """A single choice in a chat completion."""
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletion(BaseModel):
    """A chat completion response."""
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    service_tier: Optional[str] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """A single choice in a streamed chat completion chunk."""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
    logprobs: Optional[Dict[str, Any]] = None


class ChatCompletionChunk(BaseModel):
    """A streamed chat completion chunk."""
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[Usage] = None


class Model(BaseModel):
    """A model object."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    """A list of models."""
    object: Literal["list"] = "list"
    data: List[Model]