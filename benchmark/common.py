from typing import Any, Callable, List, Optional
from datetime import datetime

def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")


class ResponderWithRetries:
    """
    Raise request to LLM with 3 retries
    """

    def __init__(self, runnable_func: Callable):
        """
        Args:
        runnable: LLM agent
        validator: used to validate response
        """
        self.runnable = runnable_func

    def respond(self, args: Optional[List[Any]]=None):
        """
        Invoke LLM agent, this function will be called by LangGraph
        Args:
        state List[BaseMessage]: message history
        """
        response = []
        for attempt in range(3):
            try:
                response = self.runnable() if args == None else self.runnable(args)
                return response
            except Exception as e:
                print(str(e))
        return response

