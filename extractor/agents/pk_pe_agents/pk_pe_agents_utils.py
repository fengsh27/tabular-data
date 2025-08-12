

def format_source_tables(source_tables: list[str] | str) -> str:
    if isinstance(source_tables, list):
        return "\n".join(source_tables) if len(source_tables) > 0 else "N / A"
    else:
        return source_tables if source_tables is not None else "N / A"



