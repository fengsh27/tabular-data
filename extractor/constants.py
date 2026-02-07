from enum import Enum

cookies = {"cookie_name": "cookie_value"}
headers = {
    "authority": "www.google.com",
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
    "accept-language": "en-US,en;q=0.9",
    "cache-control": "max-age=0",
    "cookie": "SID=ZAjX93QUU1NMI2Ztt_dmL9YRSRW84IvHQwRrSe1lYhIZncwY4QYs0J60X1WvNumDBjmqCA.; __Secure-",
    # ..,
    "sec-ch-ua": '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
    "sec-ch-ua-arch": '"x86"',
    "sec-ch-ua-bitness": '"64"',
    "sec-ch-ua-full-version": '"115.0.5790.110"',
    "sec-ch-ua-full-version-list": '"Not/A)Brand";v="99.0.0.0", "Google Chrome";v="115.0.5790.110", "Chromium";v="115.0.5790.110"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-model": '""',
    "sec-ch-ua-platform": "Windows",
    "sec-ch-ua-platform-version": "15.0.0",
    "sec-ch-ua-wow64": "?0",
    "sec-fetch-dest": "document",
    "sec-fetch-mode": "navigate",
    "sec-fetch-site": "same-origin",
    "sec-fetch-user": "?1",
    "upgrade-insecure-requests": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
    "x-client-data": "#..",
}

FULL_TEXT_LENGTH_THRESHOLD = 10000  # we assume the length of full-text paper should be
# greater than 10000
MAX_FULL_TEXT_LENGTH = 31 * 1024  # should not be greater than 31K

ERROR_UNKNOWN_ERROR = -1
ERROR_OK = 5000

# PROMPTS_NAME_PK_SUM = "Pharmacokinetics Parameter Summary"
# PROMPTS_NAME_PK_IND = "Pharmacokinetics Parameter Individual"
# PROMPTS_NAME_PK_SPEC_SUM = "Pharmacokinetics Specimen Summary"
# PROMPTS_NAME_PK_DRUG_SUM = "Pharmacokinetics Drug Summary"
# PROMPTS_NAME_PK_POPU_SUM = "Pharmacokinetics Population Summary"
# PROMPTS_NAME_PK_SPEC_IND = "Pharmacokinetics Specimen Individual"
# PROMPTS_NAME_PK_DRUG_IND = "Pharmacokinetics Drug Individual"
# PROMPTS_NAME_PK_POPU_IND = "Pharmacokinetics Population Individual"
# PROMPTS_NAME_PE = "Pharmaco-Epidemiology "

class PKPEPipelineType(Enum):
    PK_SUMMARY = "PK Parameter Summary"
    PK_INDIVIDUAL = "PK Parameter Individual"
    PK_SPEC_SUMMARY = "PK Specimen Summary"
    PK_DRUG_SUMMARY = "PK Drug Summary"
    PK_POPU_SUMMARY = "PK Population Summary"
    PK_SPEC_INDIVIDUAL = "PK Specimen Individual"
    PK_DRUG_INDIVIDUAL = "PK Drug Individual"
    PK_POPU_INDIVIDUAL = "PK Population Individual"
    PE_STUDY_INFO = "PE/CT Study Information"
    PE_STUDY_OUTCOME = "PE/CT Study Outcome"

PROMPTS_NAME_PK_SUM = "PK Parameter Summary"
PROMPTS_NAME_PK_IND = "PK Parameter Individual"
PROMPTS_NAME_PK_SPEC_SUM = "PK Specimen Summary"
PROMPTS_NAME_PK_DRUG_SUM = "PK Drug Summary"
PROMPTS_NAME_PK_POPU_SUM = "PK Population Summary"
PROMPTS_NAME_PK_SPEC_IND = "PK Specimen Individual"
PROMPTS_NAME_PK_DRUG_IND = "PK Drug Individual"
PROMPTS_NAME_PK_POPU_IND = "PK Population Individual"
PROMPTS_NAME_PE_STUDY_INFO = "PE/CT Study Information"
PROMPTS_NAME_PE_STUDY_OUT = "PE/CT Study Outcome"

LLM_CHATGPT = "ChatGPT-5"
LLM_GEMINI_PRO = "Gemini 2.0 pro"
LLM_DEEPSEEK_CHAT = "DeepSeek-v3"

TABLE_ROLE_PROMPTS = "Please act as a biomedical assistant, help to extract the information from the provided source"
TABLE_SOURCE_PROMPTS = "html table from biomedical article"
PKSUMMARY_TABLE_OUTPUT_COLUMNS = [
    "DN",
    "Ana",
    "Sp",
    "Pop",
    "PS",
    "SN",
    "PT",
    "V",
    "U",
    "SS",
    "VT",
    "VV",
    "IT",
    "LL",
    "HL",
    "PV",
]
PKSUMMARY_TABLE_OUTPUT_COLUMNS_DEFINITION = [
    "Drug name: is text, the name of drug mentioned in the paper",
    "Analyte: is text, either the primary drug, its metabolite, or another drug on which the primary drug acts.",
    "Specimen: is text, what is the specimen, like 'blood', 'breast milk', 'cord blood', and so on.",
    "Pregnancy stage: is text, What pregnancy stages of patients mentioned in the paper, like 'postpartum', 'before pregnancy', '1st trimester' and so on. If not mentioned, please label as 'N/A',",
    "Parameter type: is text, the type of parameter, like 'concentration after the first dose', 'concentration after the second dose', 'clearance', 'Total area under curve' and so on.",
    "Value: is a number, the value of parameter",
    "Unit: the unit of the value",
    "Summary statistics: the statistics method to summary the data, like 'geometric mean', 'arithmetic mean'",
    "Interval type: is text, specifies the type of interval that is being used to describe uncertainty or variability around a measure or estimate, like '95% CI', 'range' and so on.",
    "Lower limit: is a number, the lower bounds of the interval",
    "Population: Describe the patient age distribution, including categories such as 'pediatric,' 'adults,' 'old adults,' 'maternal,' 'fetal,' 'neonate,' etc.",
    "High limit: is a number, the higher bounds of the interval",
    "Subject N:  the number of subjects that correspond to the specific parameter. ",
    "Variation value: is a number, the number that corresponds to the specific variation.",
    "Variation type: the variability measure (describes how spread out the data is) associated with the specific parameter, e.g., standard deviation (SD), CV%.",
    "P value: The p-value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported.",
]
TABLE_OUTPUT_NOTES = [
    "1. Only output table in json format without any other characters, no triple backticks ``` and no 'json'.",
    "2. Ensure to extract all available information for each field without omitting any details.",
    "3. If the information that is not provided, please leave it with empty string.",
]

COT_USER_INSTRUCTION = "Before jump to the final answer, provide your **detailed explanation** in your answer. \nNow, please provide **the final answer** and **your detailed explanation**."

PROMPTS_NAME_PE = "deprecate"  # "Pharmaco-Epidemiology"

MAX_STEP_COUNT = 3 * 3 # 3 agent and max 3 loops

MAX_AGENTTOOL_TASK_STEP_COUNT = 2 * 3 - 1 # 2 agent and max 3 loops

class PipelineTypeEnum(Enum):
    PK_SUMMARY = "pk_summary"
    PK_INDIVIDUAL = "pk_individual"
    PK_SPEC_SUMMARY = "pk_specimen_summary"
    PK_DRUG_SUMMARY = "pk_drug_summary"
    PK_POPU_SUMMARY = "pk_population_summary"
    PK_SPEC_INDIVIDUAL = "pk_specimen_individual"
    PK_DRUG_INDIVIDUAL = "pk_drug_individual"
    PK_POPU_INDIVIDUAL = "pk_population_individual"
    PE_STUDY_INFO = "pe_study_info"
    PE_STUDY_OUTCOME = "pe_study_outcome"

