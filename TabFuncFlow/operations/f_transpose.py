from TabFuncFlow.utils.table_utils import *


def f_transpose(df_table):
    md_table = dataframe_to_markdown(df_table)
    t_md_table = deduplicate_headers(fill_empty_headers(remove_empty_col_row(transpose_markdown_table(md_table))))
    return markdown_to_dataframe(t_md_table)