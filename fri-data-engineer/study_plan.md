# Transition Plan: HPC to Data Engineering (FRI Capstone)

## Overview
This repository tracks a 6-to-9-month study plan to transition from an HPC Scientific Application Engineer to a Modern Data Stack Data Engineer. The ultimate goal is to build an end-to-end data platform that mirrors the daily operations of the Forecasting Research Institute (FRI), specifically focusing on their ForecastBench open-source benchmark.

## The Capstone Project: A Local "ForecastBench" Data Platform
This project demonstrates the ability to ingest, model, and serve forecasting data comparing Large Language Models against human superforecasters. 

### Architecture Specifications
* **Extract:** Python scripts pull raw JSON data from the ForecastBench GitHub repository or live platforms (e.g., Metaculus API).
* **Load:** Raw data is loaded into a local PostgreSQL database running via Docker.
* **Transform:** dbt (Data Build Tool) transforms the raw JSON into a Kimball Star Schema.
* **Orchestrate:** Apache Airflow schedules daily extraction and dbt transformation runs.
* **Serve:** A local BI tool (like Metabase) connects to the Postgres database to visualize a leaderboard dashboard.

---

## Phase 1: Dimensional Data Modeling (Months 1 - 2)

* **Objective:** Learn to design a dimensional data model including staging, dimension, fact, and mart layers.
* **Core Concept:** The Kimball Methodology (Star Schemas, Fact Tables, Dimension Tables).
* **Resource 1 (Book):** *The Data Warehouse Toolkit (3rd Edition)* by Ralph Kimball and Margy Ross (Focus on Chapters 1-3).
* **Resource 2 (Book):** *Fundamentals of Data Engineering* by Joe Reis and Matt Housley.
* **Action Item:** Design the schema for the Capstone Project on paper, explicitly identifying "Facts" (e.g., individual forecast predictions) and "Dimensions" (e.g., LLM model, question category, date).

---

## Phase 2: Transformations with dbt (Month 3)

* **Objective:** Master data transformation within a warehouse using dbt.
* **Core Concept:** dbt models, jinja templating, testing, and documentation.
* **Resource 1 (Course):** The official dbt Fundamentals Course (Free).
* **Resource 2 (YouTube):** "dbt tutorial for beginners" by Kahan Data Solutions.
* **Action Item:** Install dbt-core locally and write the SQL to transform the raw ForecastBench data into the designed Star Schema.

---

## Phase 3: Orchestration and Containerization (Months 4 - 5)

* **Objective:** Apply existing containerization knowledge to Apache Airflow for data pipeline orchestration.
* **Core Concept:** Directed Acyclic Graphs (DAGs), Airflow Operators, and Docker Compose networking.
* **Resource 1 (YouTube):** DataTalks.Club - Data Engineering Zoomcamp (Modules on Docker, Postgres, and Airflow).
* **Resource 2 (YouTube):** Marc Lamberti's Airflow tutorials.
* **Action Item:** Create a `docker-compose.yml` file to spin up Airflow, PostgreSQL, and the Python environment.
* **Action Item:** Write an Airflow DAG that triggers the Python extraction script, followed by the dbt run.

---

## Phase 4: Cloud Migration & Portfolio Publication (Months 6 - 8)

* **Objective:** Prove the ability to operate production systems on a major cloud platform.
* **Core Concept:** Cloud Data Warehousing and Infrastructure as Code.
* **Action Item:** Set up a free Google Cloud Platform (GCP) account.
* **Action Item:** Migrate the local Postgres database to Google BigQuery and update the dbt profile accordingly.
* **Action Item:** Write and publish a comprehensive blog post titled "Building a Modern Data Stack for AI Forecasting: A ForecastBench Case Study."
