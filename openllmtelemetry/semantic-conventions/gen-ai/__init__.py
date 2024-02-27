from enum import Enum


class SpanAttributes:
    # LLM - Many of these will be prefixed with gen_ai
    LLM_VENDOR = "llm.vendor"
    LLM_REQUEST_TYPE = "llm.request.type"
    LLM_REQUEST_MODEL = "llm.request.model"
    LLM_RESPONSE_MODEL = "llm.response.model"
    LLM_REQUEST_MAX_TOKENS = "llm.request.max_tokens"
    LLM_USAGE_TOTAL_TOKENS = "llm.usage.total_tokens"
    LLM_USAGE_COMPLETION_TOKENS = "llm.usage.completion_tokens"
    LLM_USAGE_PROMPT_TOKENS = "llm.usage.prompt_tokens"
    LLM_TEMPERATURE = "llm.temperature"
    LLM_USER = "llm.user"
    LLM_HEADERS = "llm.headers"
    LLM_TOP_P = "llm.top_p"
    LLM_TOP_K = "llm.top_k"
    LLM_FREQUENCY_PENALTY = "llm.frequency_penalty"
    LLM_PRESENCE_PENALTY = "llm.presence_penalty"
    LLM_PROMPTS = "llm.prompts"
    LLM_COMPLETIONS = "llm.completions"
    LLM_CHAT_STOP_SEQUENCES = "llm.chat.stop_sequences"
    LLM_REQUEST_FUNCTIONS = "llm.request.functions"

    # Vector DB
    VECTOR_DB_VENDOR = "vector_db.vendor"
    VECTOR_DB_QUERY_TOP_K = "vector_db.query.top_k"


class Events(Enum):
    VECTOR_DB_QUERY_EMBEDDINGS = "vector_db.query.embeddings"
    VECTOR_DB_QUERY_RESULT = "vector_db.query.result"


class EventAttributes(Enum):
    # Query Embeddings
    VECTOR_DB_QUERY_EMBEDDINGS_VECTOR = "vector_db.query.embeddings.{i}.vector"

    # Query Result
    VECTOR_DB_QUERY_RESULT_IDS = "vector_db.query.result.{i}.ids"
    VECTOR_DB_QUERY_RESULT_DISTANCES = "vector_db.query.result.{i}.distances"
    VECTOR_DB_QUERY_RESULT_METADATA = "vector_db.query.result.{i}.metadata"
    VECTOR_DB_QUERY_RESULT_DOCUMENTS = "vector_db.query.result.{i}.documents"


class LLMRequestTypeValues(Enum):
    COMPLETION = "completion"
    CHAT = "chat"
    RERANK = "rerank"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"
