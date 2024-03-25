
import pytest
from extractor.request_geminiai import convert_messages

messages_1 = [
    {'role': "system", "content": "Hi"},
    {'role': 'user', "content": "Morning"},
    {'role': 'assistant', 'content': "Nice to meet you"}
]

messages_2 = []

messages_3 = [
    {'role': "user", "content": "Hi"},
    {'role': "assistant", "content": "Hi"},
    {'role': "user", "content": "Morning"},
    {'role': "assistant", "content": "Mornning"},
]

@pytest.mark.parametrize('messages, expected_length', [
    (messages_1, 2), 
    (messages_2, 0), 
    (messages_3, 4)
])
def test_convert_messages(messages, expected_length):
    msgs = convert_messages(messages)
    assert len(msgs) == expected_length
