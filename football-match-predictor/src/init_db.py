import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()


def build_server_url() -> str:
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "3306")
    db_user = os.getenv("DB_USER", "root")
    db_password = quote_plus(os.getenv("DB_PASSWORD", ""))

    return f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/"


def run_schema() -> None:
    project_root = Path(__file__).resolve().parents[1]
    schema_path = project_root / "sql" / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")

    statements = [statement.strip() for statement in schema_sql.split(";") if statement.strip()]

    engine = create_engine(build_server_url(), future=True)
    with engine.connect() as connection:
        for statement in statements:
            connection.execute(text(statement))
        connection.commit()

    print("Database schema initialized.")


if __name__ == "__main__":
    run_schema()
