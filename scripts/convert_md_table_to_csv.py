from pathlib import Path
import argparse
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from TabFuncFlow.utils.table_utils import markdown_to_dataframe

parser = argparse.ArgumentParser()
parser.add_argument('md_table_fn')

args = parser.parse_args()
args = vars(args)
md_table_fn = args.get("md_table_fn")
fn = Path(md_table_fn)
csv_fn = fn.with_suffix(".csv")
df = markdown_to_dataframe(fn.read_text())
df.to_csv(csv_fn, index=False)

