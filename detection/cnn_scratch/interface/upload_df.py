from google.cloud import bigquery
import pandas as pd

PROJECT = ""
DATASET = "taxifare_gulfairus"
TABLE = "lecture_data"

table = f"{PROJECT}.{DATASET}.{TABLE}"

df = pd.DataFrame({'col1': [1, 2], 'col2':[3, 4]})
client = bigquery.Client()

write_mode = "WRITE_TRUNCATE"
job_config = bigquery.LoadJobConfig()
