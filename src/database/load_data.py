import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


class LoadData:
    _engine = None

    @classmethod
    def get_engine(cls):
        if cls._engine is None:
            database_url = os.getenv("DATABASE_URL")

            if not database_url:
                raise ValueError("DATABASE_URL não encontrada.")

            cls._engine = create_engine(database_url)

        return cls._engine

    @classmethod
    def query(cls, sql: str) -> pd.DataFrame:
        return pd.read_sql(sql, cls.get_engine())
    
    @classmethod
    def load_meteorological_data(cls):
        sql = "SELECT * FROM meteorological_data"
        return cls.query(sql)
    
    
df = LoadData.load_meteorological_data()
print(df)