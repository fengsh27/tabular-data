from enum import Enum

BASELINE = "baseline"


class BenchmarkType(Enum):
    UNKNOWN = "unknown"
    PK_SUMMARY = "pk-summary"
    PK_SUMMARY_BASELINE = "pk-summary-baseline"
    PE = "pe"
    PE_BASELINE = "pe-baseline"


class LLModelType(Enum):
    GPT4O = "gpt4o"
    GEMINI15 = "gemini15"
    SONNET4 = "sonnet4"
    BASELINE = "baseline"
    UNKNOWN = "unkown"
