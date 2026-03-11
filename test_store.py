import os
from langgraph.store.postgres import PostgresStore
from dotenv import load_dotenv

load_dotenv()

POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
DB_PORT = os.getenv("DB_PORT")

DB_URI = os.getenv("DATABASE_URL", f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{DB_PORT}/{POSTGRES_DB}")

with PostgresStore.from_conn_string(DB_URI) as store:
    ns = ("user", "12345", "details")
    items = store.search(ns)

for it in items:
    print(it.value["data"])