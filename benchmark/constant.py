from enum import Enum

BASELINE = "baseline"


class BenchmarkType(Enum):
    UNKNOWN = "unknown"
    PK_SUMMARY = "pk-summary"
    PK_INDIVIDUAL = "pk-individual"
    PE_STUDY_OUTCOME = "pe-study-outcome"
    PE_STUDY_INFO = "pe-study-info"
    PK_DRUG_SUMMARY = "pk-drug-summary"
    PK_DRUG_INDIVIDUAL = "pk-drug-individual"
    PK_SPECIMEN_INDIVIDUAL = "pk-specimen-individual"
    PK_SPECIMEN_SUMMARY = "pk-specimen-summary"
    PK_POPULATION_SUMMARY = "pk-population-summary"
    PK_POPULATION_INDIVIDUAL = "pk-population-individual"
    PK_PE = "pk-pe"



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
    QWEN36SKILL="qwen36-skill"
    MAS_GPT4O_GPT54="mas-gpt4o-gpt54"  # multi-agent: pipeline llm gpt-4o, agent llm gpt-5.4
    MAS_GPT54="mas-gpt54"  # multi-agent: pipeline llm and agent llm gpt-5.4
    MAS_QWEN36="mas-qwen36"  # multi-agent: pipeline llm and agent llm qwen3.6, verify_scope=combined
    MAS_QWEN36_PT="mas-qwen36-pt"  # same as MAS_QWEN36 but verify_scope=per_table (default since ab29b0b's follow-up)
    MAS_GEMMA4="mas-gemma4"  # multi-agent: pipeline llm and agent llm gemma4:31b (verify_scope=per_table default)
    QWEN38SKILL="qwen38-skill"  # qwen3.8 (27B dense) through Claude Code + skills
    QWEN36="qwen36"  # qwen3.6 through the multi-agent pipeline (not the skills)
    QWEN38="qwen38"  # qwen3.8 (27B dense) through simple-prompt/pipeline (not the skills)
    MAS_QWEN38="mas-qwen38"  # multi-agent: pipeline llm and agent llm qwen3.8:27b, verify_scope=per_table (default)
