cookies = {
    'cookie_name': 'cookie_value'
}
headers = {
    'authority': 'www.google.com',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-language': 'en-US,en;q=0.9',
    'cache-control': 'max-age=0',
    'cookie': "SID=ZAjX93QUU1NMI2Ztt_dmL9YRSRW84IvHQwRrSe1lYhIZncwY4QYs0J60X1WvNumDBjmqCA.; __Secure-", 
    #..,
    'sec-ch-ua': '"Not/A)Brand";v="99", "Google Chrome";v="115", "Chromium";v="115"',
    'sec-ch-ua-arch': '"x86"',
    'sec-ch-ua-bitness': '"64"',
    'sec-ch-ua-full-version': '"115.0.5790.110"',
    'sec-ch-ua-full-version-list': '"Not/A)Brand";v="99.0.0.0", "Google Chrome";v="115.0.5790.110", "Chromium";v="115.0.5790.110"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-model': '""',
    'sec-ch-ua-platform': 'Windows',
    'sec-ch-ua-platform-version': '15.0.0',
    'sec-ch-ua-wow64': '?0',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'x-client-data': '#..',
}

FULL_TEXT_LENGTH_THRESHOLD = 10000 # we assume the length of full-text paper should be 
                                   # greater than 10000
MAX_FULL_TEXT_LENGTH = 31 * 1024   # should not be greater than 31K


ERROR_UNKNOWN_ERROR = -1
ERROR_OK = 5000

PROMPTS_NAME_PK = "PK Prompts"
PROMPTS_NAME_PE = "PE Prompts"

PROMPTS_NAME_PK_CHAIN = "PK Prompt Chain"
PROMPTS_NAME_PK_COT = "PK COT Prompts"

LLM_CHATGPT_4O = "ChatGPT 4o"
LLM_CHATGPT_35 = "ChatGPT 3.5-turbo"
LLM_CHATGPT_40 = "ChatGPT 4"
LLM_GEMINI_PRO = "Gemini 1.5 pro"
LLM_GEMINI_FLASH = "Gemini 1.5 flash"


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
    "P value: The p-value is a number, calculated from a statistical test, that describes the likelihood of a particular set of observations if the null hypothesis were true; varies depending on the study, and therefore it may not always be reported."
]
TABLE_OUTPUT_NOTES = [
    "1. Only output table in json format without any other characters, no triple backticks ``` and no 'json'.",
    "2. Ensure to extract all available information for each field without omitting any details.",
    "3. If the information that is not provided, please leave it with empty string."
]