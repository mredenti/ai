import json 
from pathlib import Path
from typing import NamedTuple, Optional, Iterable, Protocol
import psycopg2
import os
from itertools import islice

def batch_iterator(iterable, batch_size):
    """Yield successive n-sized chunks from an iterable."""
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch

# Define the Schema (Using Fluent Python's suggestion of NamedTuples)
class Respondent(NamedTuple):
    id: str
    age: int
    location: Optional[str] = None
    
class SurveyResponse(NamedTuple):
    respondent: Respondent
    answers: dict 
    timestamp: str
    version: str = "v1"
    
class DataSink(Protocol):
    def save_batch(self, records: Iterable[SurveyResponse]) -> None:
        """Saves a batch of cleaned records to the destination."""
        ...
            
class PostgresSink:
    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        
    def save_batch(self, records: Iterable[SurveyResponse]) -> None:
        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO staging_survey_responses
                    (respondent_id, age, location, answers, raw_version)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (respondent_id) DO NOTHING;
                    """,
                    [
                        (r.respondent.id, r.respondent.age, r.respondent.location,
                        json.dumps(r.answers), r.version)
                        for r in records
                    ]
                )
    
class SurveyIngestor:
    def __init__(self, raw_data_path: Path):
        self.raw_data_path = raw_data_path
        
    def stream_raw_records(self):
        """Generator: Efficiently yields records one by one from the raw data file."""
        with open(self.raw_data_path, 'r') as f:
            data = json.load(f)
            for record in data:
                yield record
                
    def parse_record(self, raw_record: dict) -> SurveyResponse:
        """Pattern matching for robust ingestion"""
        match raw_record:
            case {"id": rid, "metadata": {"age": a, "loc": l}, "responses": ans}:
                return SurveyResponse(Respondent(rid, a, l), ans, "N/A", "v1")
            case {"respondent_id": rid, "age": a, "location": l, "answers": ans, "ts": ts}:
                return SurveyResponse(Respondent(rid, a, l), ans, ts, "v2")
            case _:
                raise ValueError(f"Unrecognized record format: {raw_record}")
            
if __name__ == "__main__":
    # 1. Setup Connection 🔌
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_name = os.getenv("DB_NAME")
    db_host = os.getenv("DB_HOST")
    
    conn_str = f"dbname={db_name} user={db_user} password={db_pass} host={db_host}"
    
    # 2. Initialize our Components 🧩
    DATA_FILE = Path("/raw_data/raw_surveys.json")
    ingestor = SurveyIngestor(DATA_FILE)
    sink = PostgresSink(conn_str)
    
    # 3. Run the Batched Pipeline 🏎️
    # We transform our raw records into SurveyResponse objects first
    responses = (ingestor.parse_record(raw) for raw in ingestor.stream_raw_records())
    
    # Then we batch them in groups of 100
    for batch in batch_iterator(responses, batch_size=100):
        sink.save_batch(batch)
        print(f"Successfully saved a batch of {len(batch)} records.")
        