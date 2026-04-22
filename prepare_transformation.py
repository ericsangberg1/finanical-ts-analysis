import duckdb
from pathlib import Path
import re 
import pandas as pd

CSV_PATH = Path("spiff_data-2.csv")
DB_PATH =Path("spiff_data.db")
TABLE_NAME = "spiff"

def to_sql_name(name: str) -> str:
    """
    for regular expression-based column name cleaning
    E.g., Total Sales -> total_sales
    """
    cleaned = re.sub(r"\W+", "_", name.strip().lower()).strip("_")
    return cleaned or "col"
re.sub

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Drop accidental index-like export columns.
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)

    # Make column names SQL-safe and unique.
    renamed = []
    seen: dict[str, int] = {}
    for col in df.columns:
        base = to_sql_name(col)
        seen[base] = seen.get(base, 0) + 1
        renamed.append(base if seen[base] == 1 else f"{base}_{seen[base]}")
    df.columns = renamed

    # Coerces values to numeric (with day as nullable Int64)
    for col in df.columns:
        if col == "day":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {CSV_PATH}")
    
    DB_PATH = "spiff_data.db"
    
    df = pd.read_csv(CSV_PATH)
    prepared = prepare_dataframe(df)

    with duckdb.connect(DB_PATH) as conn:
        conn.register("prepared_df", prepared)
        conn.execute(f"CREATE TABLE IF NOT EXISTS {TABLE_NAME} AS SELECT * FROM prepared_df ORDER BY day")
        print("Data loaded into DuckDB:")
        print(conn.execute(f"SUMMARIZE {TABLE_NAME}").df())

        trans_sql = f"""CREATE OR REPLACE TABLE transformed_data AS
        SELECT 
        day, log( COLUMNS(* EXCLUDE day) /lag(COLUMNS(* EXCLUDE day)) OVER (ORDER BY day)) AS "\\0_logreturn"
        FROM {TABLE_NAME}
        QUALIFY lag(day) OVER (ORDER BY day) IS NOT NULL;
        """
        conn.execute(trans_sql)
        print("Data transformed with log returns:")
        print(conn.execute("SUMMARIZE transformed_data").df())


if __name__ == "__main__":
    main()