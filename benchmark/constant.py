from enum import Enum

BASELINE = "baseline"


class BenchmarkType(Enum):
    UNKNOWN = "unknown"
    PK_SUMMARY = "pk-summary"
    PK_SUMMARY_BASELINE = "pk-summary-baseline"
    PK_INDIVIDUAL = "pk-individual"
    PK_INDIVIDUAL_BASELINE = "pk-individual-baseline"
    PE = "pe"
    PE_BASELINE = "pe-baseline"


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
    UNKNOWN = "unkown"
    GPTOSS="gpt-oss"
    QWEN3="qwen3"
    CODEX="codex"
