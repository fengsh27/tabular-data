def f_split_by_rows(row_groups, df_table):
    subtables = []

    if row_groups is None:
        row_groups = [list(range(df_table.shape[0]))]

    all_rows = set(range(df_table.shape[0]))
    grouped_rows = set(sum(row_groups, []))
    if all_rows != grouped_rows:
        missing_rows = all_rows - grouped_rows
        raise ValueError(f"Missing Rows: {missing_rows}")

    for i, rows in enumerate(row_groups):
        sub_df = df_table.iloc[rows].reset_index(drop=True)
        subtables.append(sub_df)

    return subtables
