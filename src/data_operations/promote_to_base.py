from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import os

load_dotenv()

def get_engine():
    engine_url = (
        f"postgresql+psycopg2://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(engine_url)

def main():
    engine = get_engine()
    sql_path = Path("sql/transforms/01_load_base_tables.sql")
    sql_text = sql_path.read_text()

    with engine.begin() as conn:
        conn.execute(text(sql_text))

    print("Promoted staging data into base tables.")


if __name__ == "__main__":
    main()