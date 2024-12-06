import sys
sys.path.append("..")

from db.duckdb.duckdbhelper import DuckDBDatabaseHelper
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from pprint import pprint

db = DuckDBDatabaseHelper("../meters.db")

def perform_kmeans(year, num_of_clusters):
    sql_query ="""

    SELECT SUM(energy_sum),LCLid
    FROM  meters m 
    WHERE 
    date_part('year', day) = ?
    GROUP BY LCLid
    """
    
    db.connect()
    records = db.fetch_all(sql_query,
                    [year])
    db.close_connection()

    if records is None:
        return None
    
    if len(records) == 0:
        return None,None

    energy_sum = pd.DataFrame(records,columns= ["energy_sum",
                                            "LCLid"]).fillna(0)
    n_energy_sum = np.array(energy_sum["energy_sum"])


    kmeans = KMeans(n_clusters=num_of_clusters, 
                random_state=0,
                  n_init="auto").fit(n_energy_sum.reshape(-1, 1))
    labels = list(kmeans.labels_)

    clusters = pd.DataFrame()
    clusters["energy_sum"] = energy_sum["energy_sum"]
    clusters["labels"] = labels

    result = clusters.groupby('labels')['energy_sum'].agg(['median', 'count'])
    result = result.sort_values(by="count",ascending=False)

    return clusters,result
