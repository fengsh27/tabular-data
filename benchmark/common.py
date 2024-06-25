
from datetime import datetime

def output_msg(msg: str):
    with open("./benchmark-result.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()}: \n{msg}\n")

