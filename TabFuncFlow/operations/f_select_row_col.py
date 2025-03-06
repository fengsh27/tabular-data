def f_select_row_col(row_list, col_list, df_table):
    try:
        return df_table.iloc[row_list][col_list].reset_index(drop=True)
    except Exception:
        return df_table