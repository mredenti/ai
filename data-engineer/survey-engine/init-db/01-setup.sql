CREATE TABLE IF NOT EXISTS staging_survey_responses (
    respondent_id TEXT PRIMARY KEY,
    age INTEGER,
    location TEXT,
    answers JSONB,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    raw_version TEXT
);