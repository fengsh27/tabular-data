
# from dotenv import load_dotenv
# load_dotenv()


def get_llm_response(messages, question, model="gemini_15_pro"):
    """
    A further wrapper around Shaohong's request_llm function.
    Send messages and question to the specified LLM and return response details.
    """

    prompt_list = [{"role": "user", "content": msg} for msg in messages]

    if model == "chatgpt_4o":
        # request_llm = request_to_chatgpt_4o
        return None, None, None, None
    elif model == "gemini_15_pro":
        # request_llm = request_to_gemini_15_pro
        return None, None, None, None
    else:
        raise ValueError(f"Unsupported model: {model}")


