def f_select_row_col(row_list, col_list, df_table):
    try:
        row_list = row_list if row_list else slice(None)
        col_list = col_list if col_list else df_table.columns
        return df_table.iloc[row_list][col_list].reset_index(drop=True)
    except Exception as e:
        print(f"Error: {e}")
        return df_table
