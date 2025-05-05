---
jupyter:
  kernelspec:
    display_name: zrive-ds-yVHW3Yen-py3.11
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.11.0
  nbformat: 4
  nbformat_minor: 5
---

::: {#8db7c8d2 .cell .markdown}
# Module 2 Task 2: Exploratory Data Analysis - Box Builder Dataset

This notebook contains the exploratory data analysis of the
`sampled_box_builder_df.csv` dataset, where each row represents an
(order, product) pair with an outcome variable indicating whether the
product was purchased or not.
:::

::: {#44ecb643 .cell .markdown}
## 1. Initial Setup and Configuration {#1-initial-setup-and-configuration}
:::

::: {#afa9c569 .cell .code execution_count="2"}
``` python
# Required imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
from dotenv import load_dotenv
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

load_dotenv()
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
```
:::

::: {#6f89f54a .cell .markdown}
## 2. Data Loading {#2-data-loading}
:::

::: {#e44a4d42 .cell .code execution_count="10"}
``` python
S3_PATH = "s3://zrive-ds-data/groceries/box_builder_dataset/feature_frame.csv"
LOCAL_DATA_PATH = 'box_builder_data/'
LOCAL_FILE_PATH = LOCAL_DATA_PATH + 'feature_frame.csv'

def download_box_builder_data():
    """Download box builder dataset from S3 if not exists locally."""
    os.makedirs(LOCAL_DATA_PATH, exist_ok=True)
    
    if os.path.exists(LOCAL_FILE_PATH):
        print(f"File already exists at {LOCAL_FILE_PATH}")
        return
    
    try:
        s3 = boto3.client('s3',
                         aws_access_key_id=AWS_ACCESS_KEY_ID,
                         aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        
        bucket_name = 'zrive-ds-data'
        key = 'groceries/box_builder_dataset/feature_frame.csv'
        
        print(f"Downloading from {S3_PATH}")
        s3.download_file(bucket_name, key, LOCAL_FILE_PATH)
        print(f"File downloaded successfully to {LOCAL_FILE_PATH}")
        
    except Exception as e:
        print(f"Error downloading file: {e}")

# Download data if needed
download_box_builder_data()

# Load the dataset
box_builder_df = pd.read_csv(LOCAL_FILE_PATH)
print(f"Dataset shape: {box_builder_df.shape}")
```

::: {.output .stream .stdout}
    Downloading from s3://zrive-ds-data/groceries/box_builder_dataset/feature_frame.csv
    File downloaded successfully to box_builder_data/feature_frame.csv
    Dataset shape: (2880549, 27)
:::
:::

::: {#4d5c524b .cell .markdown}
## 3. First Look at the Data {#3-first-look-at-the-data}
:::

::: {#1f708b3d .cell .code execution_count="13"}
``` python
box_builder_df.head()
```

::: {.output .execute_result execution_count="13"}
```{=html}
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
      <th>variant_id</th>
      <th>product_type</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>created_at</th>
      <th>order_date</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2807985930372</td>
      <td>3482464092292</td>
      <td>2020-10-05 16:46:19</td>
      <td>2020-10-05 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>1</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808027644036</td>
      <td>3466586718340</td>
      <td>2020-10-05 17:59:51</td>
      <td>2020-10-05 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>2</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808099078276</td>
      <td>3481384026244</td>
      <td>2020-10-05 20:08:53</td>
      <td>2020-10-05 00:00:00</td>
      <td>4</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>3</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808393957508</td>
      <td>3291363377284</td>
      <td>2020-10-06 08:57:59</td>
      <td>2020-10-06 00:00:00</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
    <tr>
      <th>4</th>
      <td>33826472919172</td>
      <td>ricepastapulses</td>
      <td>2808429314180</td>
      <td>3537167515780</td>
      <td>2020-10-06 10:37:05</td>
      <td>2020-10-06 00:00:00</td>
      <td>3</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>31.134053</td>
      <td>30.0</td>
      <td>30.0</td>
      <td>24.27618</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>
```
:::
:::

::: {#251845b3 .cell .code execution_count="15"}
``` python
box_builder_df.columns
```

::: {.output .execute_result execution_count="15"}
    Index(['variant_id', 'product_type', 'order_id', 'user_id', 'created_at',
           'order_date', 'user_order_seq', 'outcome', 'ordered_before',
           'abandoned_before', 'active_snoozed', 'set_as_regular',
           'normalised_price', 'discount_pct', 'vendor', 'global_popularity',
           'count_adults', 'count_children', 'count_babies', 'count_pets',
           'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')
:::
:::

::: {#7617c17d .cell .markdown}
## 4. Data Quality Checks {#4-data-quality-checks}
:::

::: {#fd8dfbb3 .cell .markdown}
### 4.1. Basic Data Validation {#41-basic-data-validation}
:::

::: {#3bf63988 .cell .code execution_count="16"}
``` python
# Check for duplicates
duplicate_count = box_builder_df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Check missing values
missing_values = box_builder_df.isnull().sum()
missing_percentage = (missing_values / len(box_builder_df)) * 100

print("\nMissing values analysis:")
for col, count in missing_values[missing_values > 0].items():
    print(f"{col}: {count} ({missing_percentage[col]:.2f}%)")
```

::: {.output .stream .stdout}
    Number of duplicate rows: 0

    Missing values analysis:
:::
:::

::: {#dabfd056 .cell .markdown}
### 4.2. Data Types and Consistency {#42-data-types-and-consistency}
:::

::: {#d6c0e464 .cell .code execution_count="17"}
``` python
# Check data types
print("Data types:")
print(box_builder_df.dtypes)
```

::: {.output .stream .stdout}
    Data types:
    variant_id                            int64
    product_type                         object
    order_id                              int64
    user_id                               int64
    created_at                           object
    order_date                           object
    user_order_seq                        int64
    outcome                             float64
    ordered_before                      float64
    abandoned_before                    float64
    active_snoozed                      float64
    set_as_regular                      float64
    normalised_price                    float64
    discount_pct                        float64
    vendor                               object
    global_popularity                   float64
    count_adults                        float64
    count_children                      float64
    count_babies                        float64
    count_pets                          float64
    people_ex_baby                      float64
    days_since_purchase_variant_id      float64
    avg_days_to_buy_variant_id          float64
    std_days_to_buy_variant_id          float64
    days_since_purchase_product_type    float64
    avg_days_to_buy_product_type        float64
    std_days_to_buy_product_type        float64
    dtype: object
:::
:::

::: {#4f110b92 .cell .code execution_count="18"}
``` python
# Check for any infinite values
inf_check = box_builder_df.replace([np.inf, -np.inf], np.nan).isnull().sum() - missing_values
print("\nInfinite values check:")
for col, count in inf_check[inf_check > 0].items():
    print(f"{col}: {count} infinite values")
```

::: {.output .stream .stdout}

    Infinite values check:
:::
:::

::: {#db091cd4 .cell .markdown}
## 5. Descriptive Statistics {#5-descriptive-statistics}
:::

::: {#07d9d8f7 .cell .markdown}
### 5.1. Numerical Features Overview {#51-numerical-features-overview}
:::

::: {#7daf733c .cell .code execution_count="24"}
``` python
numerical_cols = box_builder_df.select_dtypes(include=[np.number]).columns
print(numerical_cols)
box_builder_df[numerical_cols].describe()
```

::: {.output .stream .stdout}
    Index(['variant_id', 'order_id', 'user_id', 'user_order_seq', 'outcome',
           'ordered_before', 'abandoned_before', 'active_snoozed',
           'set_as_regular', 'normalised_price', 'discount_pct',
           'global_popularity', 'count_adults', 'count_children', 'count_babies',
           'count_pets', 'people_ex_baby', 'days_since_purchase_variant_id',
           'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
           'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
           'std_days_to_buy_product_type'],
          dtype='object')
:::

::: {.output .execute_result execution_count="24"}
```{=html}
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
      <th>variant_id</th>
      <th>order_id</th>
      <th>user_id</th>
      <th>user_order_seq</th>
      <th>outcome</th>
      <th>ordered_before</th>
      <th>abandoned_before</th>
      <th>active_snoozed</th>
      <th>set_as_regular</th>
      <th>normalised_price</th>
      <th>...</th>
      <th>count_children</th>
      <th>count_babies</th>
      <th>count_pets</th>
      <th>people_ex_baby</th>
      <th>days_since_purchase_variant_id</th>
      <th>avg_days_to_buy_variant_id</th>
      <th>std_days_to_buy_variant_id</th>
      <th>days_since_purchase_product_type</th>
      <th>avg_days_to_buy_product_type</th>
      <th>std_days_to_buy_product_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>...</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
      <td>2.880549e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.401250e+13</td>
      <td>2.978388e+12</td>
      <td>3.750025e+12</td>
      <td>3.289342e+00</td>
      <td>1.153669e-02</td>
      <td>2.113868e-02</td>
      <td>6.092589e-04</td>
      <td>2.290188e-03</td>
      <td>3.629864e-03</td>
      <td>1.272808e-01</td>
      <td>...</td>
      <td>5.492182e-02</td>
      <td>3.538562e-03</td>
      <td>5.134091e-02</td>
      <td>2.072549e+00</td>
      <td>3.312961e+01</td>
      <td>3.523734e+01</td>
      <td>2.645304e+01</td>
      <td>3.143513e+01</td>
      <td>3.088810e+01</td>
      <td>2.594969e+01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.786246e+11</td>
      <td>2.446292e+11</td>
      <td>1.775710e+11</td>
      <td>2.140176e+00</td>
      <td>1.067876e-01</td>
      <td>1.438466e-01</td>
      <td>2.467565e-02</td>
      <td>4.780109e-02</td>
      <td>6.013891e-02</td>
      <td>1.268378e-01</td>
      <td>...</td>
      <td>3.276586e-01</td>
      <td>5.938048e-02</td>
      <td>3.013646e-01</td>
      <td>3.943659e-01</td>
      <td>3.707162e+00</td>
      <td>1.057766e+01</td>
      <td>7.168323e+00</td>
      <td>1.227511e+01</td>
      <td>4.330262e+00</td>
      <td>3.278860e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.361529e+13</td>
      <td>2.807986e+12</td>
      <td>3.046041e+12</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.599349e-02</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.414214e+00</td>
      <td>0.000000e+00</td>
      <td>7.000000e+00</td>
      <td>2.828427e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.380354e+13</td>
      <td>2.875152e+12</td>
      <td>3.745901e+12</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.394416e-02</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.000000e+01</td>
      <td>2.319372e+01</td>
      <td>3.000000e+01</td>
      <td>2.800000e+01</td>
      <td>2.427618e+01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.397325e+13</td>
      <td>2.902856e+12</td>
      <td>3.812775e+12</td>
      <td>3.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>8.105178e-02</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>3.400000e+01</td>
      <td>2.769305e+01</td>
      <td>3.000000e+01</td>
      <td>3.100000e+01</td>
      <td>2.608188e+01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.428495e+13</td>
      <td>2.922034e+12</td>
      <td>3.874925e+12</td>
      <td>4.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.352670e-01</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.300000e+01</td>
      <td>4.000000e+01</td>
      <td>3.059484e+01</td>
      <td>3.000000e+01</td>
      <td>3.400000e+01</td>
      <td>2.796118e+01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.454300e+13</td>
      <td>3.643302e+12</td>
      <td>5.029635e+12</td>
      <td>2.100000e+01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>...</td>
      <td>3.000000e+00</td>
      <td>1.000000e+00</td>
      <td>6.000000e+00</td>
      <td>5.000000e+00</td>
      <td>1.480000e+02</td>
      <td>8.400000e+01</td>
      <td>5.868986e+01</td>
      <td>1.480000e+02</td>
      <td>3.950000e+01</td>
      <td>3.564191e+01</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 23 columns</p>
</div>
```
:::
:::

::: {#8b049b57 .cell .markdown}
### 5.2. Categorical Features Overview {#52-categorical-features-overview}
:::

::: {#30b96f2c .cell .code execution_count="21"}
``` python
categorical_cols = box_builder_df.select_dtypes(include=['object', 'category']).columns
print(f"Categorical columns: {categorical_cols.tolist()}")

for col in categorical_cols[:5]: 
    print(f"\n{col} value counts:")
    print(box_builder_df[col].value_counts())
```

::: {.output .stream .stdout}
    Categorical columns: ['product_type', 'created_at', 'order_date', 'vendor']

    product_type value counts:
    product_type
    tinspackagedfoods         226474
    condimentsdressings       129749
    ricepastapulses           128098
    haircare                  114978
    cookingingredientsoils    110686
                               ...  
    babyfood12months            6797
    householdsundries           6735
    petcare                     4075
    feedingweaning              2790
    premixedcocktails           2620
    Name: count, Length: 62, dtype: int64

    created_at value counts:
    created_at
    2021-03-03 14:42:05    976
    2021-03-03 11:36:16    976
    2021-03-02 22:36:33    976
    2021-03-02 23:56:57    976
    2021-03-03 00:33:51    976
                          ... 
    2020-10-06 10:50:23    614
    2020-10-06 08:57:59    611
    2020-10-05 20:08:53    608
    2020-10-05 17:59:51    608
    2020-10-05 16:46:19    608
    Name: count, Length: 3446, dtype: int64

    order_date value counts:
    order_date
    2021-02-17 00:00:00    68446
    2021-02-25 00:00:00    60488
    2021-03-01 00:00:00    58397
    2021-02-18 00:00:00    52822
    2021-02-26 00:00:00    51700
                           ...  
    2020-10-22 00:00:00     2689
    2020-10-10 00:00:00     2608
    2020-10-17 00:00:00     1996
    2020-10-05 00:00:00     1824
    2020-12-24 00:00:00      810
    Name: count, Length: 149, dtype: int64

    vendor value counts:
    vendor
    biona         146828
    ecover        113018
    method         79258
    organix        74632
    treeoflife     68920
                   ...  
    minky            599
    vitalbaby        594
    munchkin         422
    freee            336
    vicks            243
    Name: count, Length: 264, dtype: int64
:::
:::

::: {#8e14b362 .cell .markdown}
## 6. Target Variable Analysis {#6-target-variable-analysis}
:::

::: {#4c6fa194 .cell .code}
``` python
# Assuming 'outcome' is our target variable
target_column = 'outcome'

if target_column in box_builder_df.columns:
    print(f"Target variable ({target_column}) distribution:")
    print(box_builder_df[target_column].value_counts())
    print(f"\nTarget variable proportion:")
    print(box_builder_df[target_column].value_counts(normalize=True))
    
    # Visualize target distribution
    plt.figure(figsize=(8, 6))
    box_builder_df[target_column].value_counts().plot(kind='bar')
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
```

::: {.output .stream .stdout}
    Target variable (outcome) distribution:
    outcome
    0.0    2847317
    1.0      33232
    Name: count, dtype: int64

    Target variable proportion:
    outcome
    0.0    0.988463
    1.0    0.011537
    Name: proportion, dtype: float64
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/527dacf79fe53a0a89997c93f45c2c3fea5166bb.png)
:::
:::

::: {#f219f0c9 .cell .markdown}
## 7. Univariate Analysis {#7-univariate-analysis}
:::

::: {#1c4d207b .cell .markdown}
### 7.1. Numerical Feature Distributions {#71-numerical-feature-distributions}
:::

::: {#9bf5fce2 .cell .code execution_count="37"}
``` python
# Select interesting numerical columns 
cols = ['user_order_seq', 'outcome', 'normalised_price', 'discount_pct',
       'global_popularity','days_since_purchase_variant_id',
       'avg_days_to_buy_variant_id', 'std_days_to_buy_variant_id',
       'days_since_purchase_product_type', 'avg_days_to_buy_product_type',
       'std_days_to_buy_product_type']
for col in cols:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    ax1.hist(box_builder_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
    ax1.set_title(f'Distribution of {col}')
    ax1.set_xlabel(col)
    ax1.set_ylabel('Frequency')
    
    # Box plot
    ax2.boxplot(box_builder_df[col].dropna())
    ax2.set_title(f'Box Plot of {col}')
    ax2.set_ylabel(col)
    
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/2464afe5838e1d4339ee187ca0de973b04bf7d99.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/5d14a2e00356aba97e803fc87a515d58b611aea6.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/aebb559462d94b1f6eb2f83fbe23a49ae2d99543.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/a2908bd5acf94a8090979898b626bc8eca40f782.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/9eea9d401a3e297ec02fa7adb9e5da89087e4c8a.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/b084cfe58170ff18400389d36caf81e12d0c3f25.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/38e2950d9413c8e7d27ec25f8007a1a172aa2b63.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/f69f759b22e9426269e9a47be72b104993ed1d19.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/8593eea57e838265daff0f40d68f0dca86c0a7e3.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/1ebfd57ade73e95740fa17bdbeb8210cb723ec74.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/0b688e25852d640c37ff75fcab4f484b712df533.png)
:::
:::

::: {#443b2f9b .cell .markdown}
### 7.2. Categorical Feature Distributions {#72-categorical-feature-distributions}
:::

::: {#88f3062d .cell .code execution_count="32"}
``` python
# Plot distribution of categorical features
for col in categorical_cols[:3]:  # Limit to first 3
    plt.figure(figsize=(12, 6))
    value_counts = box_builder_df[col].value_counts()
    
    if len(value_counts) > 10:
        value_counts = value_counts.head(10)
        title = f'Top 10 values in {col}'
    else:
        title = f'Distribution of {col}'
    
    value_counts.plot(kind='bar')
    plt.title(title)
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
```

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/b83dc04e5df40e71ab2395f48fe952f14357c638.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/8d535b11b843665bc453d2b5f75672db5c803d3d.png)
:::

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/74c5b05d93f88a02fe01c7cea8eefae8cccde876.png)
:::
:::

::: {#6e60ec44 .cell .markdown}
## 8. Correlation Analysis {#8-correlation-analysis}
:::

::: {#a38b7714 .cell .code execution_count="35"}
``` python
# Correlation matrix for numerical features
correlation_matrix = box_builder_df[cols].corr()

# Plot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.show()
```

::: {.output .display_data}
![](vertopal_a3a184c3d05244fb8df19dd79985209c/3322ab6818fc6d13c1d11fd34d3036837b730bbf.png)
:::
:::

::: {#2b4e2b1a .cell .markdown}
# Conclusiones

1.  Dataset muy desbalanceado segun la columna target \'outcome\'

No le he dedicado mucho tiempo ya que la mayoria del tiempo lo he
perdido con el analisis de la primera tarea.
:::
