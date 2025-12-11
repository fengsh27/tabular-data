import json
from typing import Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import ChatOllama
from pydantic import BaseModel

def structured_output_llm(
    llm,
    schema: Any,
    prompt: ChatPromptTemplate,
):
    if isinstance(llm, BaseChatOpenAI):
        agent = prompt | llm.with_structured_output(schema)
        return agent
    elif isinstance(llm, ChatOllama):
        parser = PydanticOutputParser(pydantic_object=schema)
        agent = prompt | llm | parser
        return agent
    else:
        raise ValueError(f"Unsupported LLM: {llm}")

def get_format_instructions(schema: Any) -> str:
    if schema is None:
        return ""
    if isinstance(schema, dict):
        reduced_schema = schema
        if "title" in reduced_schema:
            del reduced_schema["title"]
        if "type" in reduced_schema:
            del reduced_schema["type"]
        schema_str = json.dumps(reduced_schema, ensure_ascii=False)
        # schema_str = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.
# **Do not** include this json schema in your output.
# ```
# {schema_str}
# ```
# """
        ret_prompt = f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{schema_str}
```
"""
        return ret_prompt
    elif isinstance(schema, type) and issubclass(schema, BaseModel):
        schema_str = PydanticOutputParser(pydantic_object=schema).get_format_instructions()
    else:
        raise ValueError(f"Unsupported schema: {schema}")
    return f"""---

### **Output Format Instructions**
{schema_str}
"""