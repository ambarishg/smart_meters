```python
import sys
sys.path.append("..")
```


```python
from db.duckdb.duckdbhelper import DuckDBDatabaseHelper
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
from pprint import pprint
```


```python
db = DuckDBDatabaseHelper("../meters.db")
db.connect()
```

    Connected to DuckDB database: ../meters.db
    


```python
sql_query ="""

SELECT SUM(energy_sum),LCLid
FROM  meters m 
WHERE 
date_part('year', day) = 2014
GROUP BY LCLid
"""

db.connect()
records = db.fetch_all(sql_query)
db.close_connection()
```

    Connected to DuckDB database: ../meters.db
    Fetched 5108 rows.
    Connection closed.
    


```python
energy_sum = pd.DataFrame(records,columns= ["energy_sum",
                                            "LCLid"]).fillna(0)
```


```python
n_energy_sum = np.array(energy_sum["energy_sum"])
```


```python
kmeans = KMeans(n_clusters=5, 
                random_state=0,
                  n_init="auto").fit(n_energy_sum.reshape(-1, 1))
labels = list(kmeans.labels_)
```


```python
clusters = pd.DataFrame()
```


```python
clusters["energy_sum"] = energy_sum["energy_sum"]
```


```python
clusters["labels"] = labels
```


```python
clusters
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_sum</th>
      <th>labels</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2231.721002</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>949.765999</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1036.168000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1078.617000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2399.529000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>5103</th>
      <td>1319.622000</td>
      <td>2</td>
    </tr>
    <tr>
      <th>5104</th>
      <td>500.209000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5105</th>
      <td>616.961000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5106</th>
      <td>349.471000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5107</th>
      <td>1147.008002</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>5108 rows × 2 columns</p>
</div>




```python
result = clusters.groupby('labels')['energy_sum'].agg(['median', 'count'])
```


```python
result.sort_values(by="median",ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>median</th>
      <th>count</th>
    </tr>
    <tr>
      <th>labels</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>3284.566</td>
      <td>53</td>
    </tr>
    <tr>
      <th>0</th>
      <td>1883.313</td>
      <td>239</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1087.135</td>
      <td>776</td>
    </tr>
    <tr>
      <th>4</th>
      <td>631.936</td>
      <td>1761</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300.092</td>
      <td>2279</td>
    </tr>
  </tbody>
</table>
</div>




```python
energy_sum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_sum</th>
      <th>LCLid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2231.721002</td>
      <td>MAC000778</td>
    </tr>
    <tr>
      <th>1</th>
      <td>949.765999</td>
      <td>MAC000850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1036.168000</td>
      <td>MAC002924</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1078.617000</td>
      <td>MAC002937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2399.529000</td>
      <td>MAC003166</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(energy_sum),len(clusters)
```




    (5108, 5108)




```python
energy_sum["cluster_label"] = clusters["labels"]
```


```python
energy_sum.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_sum</th>
      <th>LCLid</th>
      <th>cluster_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2231.721002</td>
      <td>MAC000778</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>949.765999</td>
      <td>MAC000850</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1036.168000</td>
      <td>MAC002924</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1078.617000</td>
      <td>MAC002937</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2399.529000</td>
      <td>MAC003166</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
energy_sum[energy_sum["cluster_label"] == 2].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_sum</th>
      <th>LCLid</th>
      <th>cluster_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>949.765999</td>
      <td>MAC000850</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1036.168000</td>
      <td>MAC002924</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1078.617000</td>
      <td>MAC002937</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>913.555000</td>
      <td>MAC001598</td>
      <td>2</td>
    </tr>
    <tr>
      <th>35</th>
      <td>989.239000</td>
      <td>MAC001371</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
energy_sum[energy_sum["cluster_label"] == 4].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>energy_sum</th>
      <th>LCLid</th>
      <th>cluster_label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>528.289</td>
      <td>MAC004529</td>
      <td>4</td>
    </tr>
    <tr>
      <th>8</th>
      <td>576.595</td>
      <td>MAC001546</td>
      <td>4</td>
    </tr>
    <tr>
      <th>14</th>
      <td>661.406</td>
      <td>MAC002069</td>
      <td>4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>872.723</td>
      <td>MAC000681</td>
      <td>4</td>
    </tr>
    <tr>
      <th>20</th>
      <td>758.442</td>
      <td>MAC000709</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
sql_query = """
DROP TABLE IF EXISTS cluster_energy 
"""
db.connect()
records = db.execute_query(sql_query)
db.close_connection()
```

    Connected to DuckDB database: ../meters.db
    Query executed successfully.
    Connection closed.
    


```python
sql_query = """
CREATE TABLE IF NOT EXISTS cluster_energy AS
SELECT * FROM energy_sum
"""
db.connect()
db.register_df("energy_sum",energy_sum)
records = db.execute_query(sql_query)
db.close_connection()
```

    Connected to DuckDB database: ../meters.db
    Query executed successfully.
    Connection closed.
    


```python
sql_query = """
SELECT * FROM cluster_energy
"""
db.connect()
records = db.fetch_all(sql_query)
db.close_connection()
```

    Connected to DuckDB database: ../meters.db
    Fetched 5108 rows.
    Connection closed.
    


```python
records[:4]
```




    [(2231.7210023, 'MAC000778', 0),
     (949.7659987, 'MAC000850', 2),
     (1036.1679999, 'MAC002924', 2),
     (1078.6169998999997, 'MAC002937', 2)]




```python
db.connect()
sql_query ="""
SELECT table_name, table_type 
FROM information_schema.tables;
"""
records = db.fetch_all(sql_query)
pprint(records)

db.close_connection()
```

    Connected to DuckDB database: ../meters.db
    Fetched 6 rows.
    [('cluster_energy', 'BASE TABLE'),
     ('info_household', 'VIEW'),
     ('meters', 'VIEW'),
     ('uk_bank_holidays', 'VIEW'),
     ('weather_daily_darksky', 'VIEW'),
     ('weather_daily_darksky_modified', 'VIEW')]
    Connection closed.
    


```python

```
