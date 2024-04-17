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

LLM_CHATGPT_35 = "ChatGPT 3.5-turbo"
LLM_GERMINI_PRO = "Germini 1.0 pro"