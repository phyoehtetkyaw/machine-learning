import pandas as pd
import numpy as np
import datetime

def prepare(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Feature Engineering
    data['age'] = datetime.datetime.now().year - data.yr_built.astype(int)

    renovated = datetime.datetime.now().year - data.yr_renovated.astype(int)
    data['renovated'] = np.where(data.yr_renovated.astype(int) > 0, renovated, 0)

    normalize_cols = ["sqft_above", "sqft_living", "sqft_lot", "sqft_basement", "age", "renovated"]
    for col in normalize_cols:
      data[col] = np.log1p(data[col].values)

    data.drop(columns=["date", "street", "city", "statezip", "country", "yr_built", "yr_renovated"], inplace=True)

    return data

def min_max_normalize(x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())

def train(x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.dot(x, w) + b

if __name__ == "__main__":
  data = {
    'date': '2018-03-15 00:00:00',
    'price': 425000.0,
    'bedrooms': 3.0,
    'bathrooms': 2.0,
    'sqft_living': 1850,
    'sqft_lot': 7200,
    'floors': 2.0,
    'waterfront': 0,
    'view': 2,
    'condition': 4,
    'sqft_above': 1850,
    'sqft_basement': 0,
    'yr_built': 1987,
    'yr_renovated': 2010,
    'street': '15432 Oak Ave',
    'city': 'Bellevue',
    'statezip': 'WA 98004',
    'country': 'USA'
  }

  df = prepare(pd.DataFrame([data]))
  df.drop(columns=["price"], inplace=True)
  x = np.array(df)
  w = np.array([ 0.06617491,  0.23069234,  0.05187711,  0.00498497,  0.39417264, -0.04178798,
        0.03783219,  0.27512106,  0.04420802,  0.05883424,  0.23939943,  0.08514919])
  b = 10.473019935695667
  

  y_predict_real = train(x, w, b)
  print(np.expm1(y_predict_real))