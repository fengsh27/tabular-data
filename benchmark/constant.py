from enum import Enum

BASELINE = "baseline"


class BenchmarkType(Enum):
    UNKNOWN = "unknown"
    PK_SUMMARY = "pk-summary"
    PK_INDIVIDUAL = "pk-individual"
    PE = "pe"


class LLModelType(Enum):
    GPT4O = "gpt4o"
    GPT5 = "gpt5"
    GEMINI15 = "gemini15"
    SONNET4 = "sonnet4"
    GEMINI25PRO = "gemini25pro"
    GEMINI25FLASH = "gemini25flash"
    GEMINI25FLASHLITE = "gemini25flashlite"
    METALLAMA4 = "metallama4"
    BASELINE = "baseline"
    UNKNOWN = "unknown"
    GPTOSS="gpt-oss"
    QWEN3="qwen3"
    CODEX="codex"
    GPT54="gpt54"
    GEMMA4="gemma4"
    QWEN35="qwen35"
