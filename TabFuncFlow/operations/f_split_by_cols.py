def f_split_by_cols(col_groups, df_table):
    subtables = []

    if col_groups is None:
        col_groups = [list(df_table.columns)]

    all_cols = set(df_table.columns)
    grouped_cols = set(sum(col_groups, []))
    if all_cols != grouped_cols:
        missing_cols = all_cols - grouped_cols
        raise ValueError(f"Missing Cols: {missing_cols}")

    for i, cols in enumerate(col_groups):
        sub_df = df_table.iloc[:][cols].reset_index(drop=True)
        subtables.append(sub_df)

    return subtables
