import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

load_dotenv()

def get_engine() -> Engine:
    engine_url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    )

    return create_engine(engine_url)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [col.lower() for col in df.columns]
    return df

def truncate_staging_tables(engine: Engine) -> None:
    staging_tables = [
        "staging_machines",
        "staging_telemetry",
        "staging_errors",
        "staging_maintenance",
        "staging_failures"
    ]

    with engine.begin() as conn:
        for table in staging_tables:
            conn.execute(text(f"TRUNCATE TABLE {table};"))
    
    print("Truncated staging tables.")

def load_csv_to_staging(csv_path: Path, table_name: str, engine: Engine) -> None:
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",
        index=False,
        method="multi",
        chunksize=10000,
    )
    print(f"Loaded {len(df):,} rows into {table_name}")

def main() -> None:
    engine = get_engine()
    data_dir = Path("data/raw")

    file_to_table = {
        "PdM_machines.csv": "staging_machines",
        "PdM_telemetry.csv": "staging_telemetry",
        "PdM_errors.csv": "staging_errors",
        "PdM_maint.csv": "staging_maintenance",
        "PdM_failures.csv": "staging_failures",
    }

    truncate_staging_tables(engine)

    for filename, table_name in file_to_table.items():
        path = data_dir / filename
        if not path.exists():
            print(f"Missing file {path}")
            continue
        
        load_csv_to_staging(path, table_name, engine)

if __name__ == "__main__":
    main()

