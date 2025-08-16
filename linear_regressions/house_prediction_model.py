import pandas as pd
import numpy as np
import datetime

class LinearRegression:
  def __init__(self):
    pass

  def load(self, path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    df = pd.DataFrame(data)
    return df

  def prepare(self, data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Feature Engineering
    data['age'] = datetime.datetime.now().year - data.yr_built.astype(int)

    renovated = datetime.datetime.now().year - data.yr_renovated.astype(int)
    data['renovated'] = np.where(data.yr_renovated.astype(int) > 0, renovated, 0)

    normalize_cols = ["sqft_above", "sqft_living", "sqft_lot", "sqft_basement", "age", "renovated"]
    for col in normalize_cols:
      data[col] = self.min_max_normalize(data[col].values)

    data.drop(columns=["date", "street", "city", "statezip", "country", "yr_built", "yr_renovated"], inplace=True)

    return data

  def min_max_normalize(self, x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())

  def separate(self, df: pd.DataFrame) -> dict[str, np.ndarray]:
    x_shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    m = len(x_shuffled_df)
    x_train = x_shuffled_df.iloc[:int(m * 0.6)].copy()
    x_val = x_shuffled_df.iloc[int(m * 0.6):int(m * 0.8)].copy()
    x_test = x_shuffled_df.iloc[int(m * 0.8):].copy()

    y_train = np.array(np.log1p(x_train.price))
    y_val = np.array(np.log1p(x_val.price))
    y_test = np.array(np.log1p(x_test.price))

    x_train.drop(columns=["price"], inplace=True)
    x_val.drop(columns=["price"], inplace=True)
    x_test.drop(columns=["price"], inplace=True)

    x_train = np.array(x_train)
    x_val = np.array(x_val)
    x_test = np.array(x_test)

    return {
      "x_train": x_train,
      "y_train": y_train,
      "x_val": x_val,
      "y_val": y_val,
      "x_test": x_test,
      "y_test": y_test
    }

  def train(self, x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return np.dot(w, x) + b

  def cost(self, x_train: np.ndarray, y_train: np.ndarray, w: np.ndarray, b: float) -> float:
    m = x_train.shape[0]

    cost = 0
    for i in range(m):
      f_x = self.train(x_train[i], w, b)
      cost += (f_x - y_train[i]) ** 2

    return cost / (2 * m)

  def gradient_descent(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, alpha: float, epochs: int):

    w_init = w
    b_init = b

    for i in range(epochs):
      dw, db = self.derivative(w_init, b_init, x, y)
      w_init = w_init - alpha * dw
      b_init = b_init - alpha * db

      if i % (epochs / 10) == 0:
        print(f"Epoch {i} - Cost: {self.cost(x, y, w_init, b_init)}")
    
    return w_init, b_init

  def derivative(self, w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray):
    dw = 0
    db = 0
    m = x.shape[0]

    for i in range(m):
      f_x = self.train(x[i], w, b)

      dw += (f_x - y[i]) * x[i]
      db += (f_x - y[i])

    return dw / m, db / m

if __name__ == "__main__":
  model = LinearRegression()
  df = model.load("data/data.csv")
  prepared_df = model.prepare(df)
  data = model.separate(prepared_df)

  w_init = np.zeros(data["x_train"].shape[1])
  b_init = 10

  # Train model
  y_predict_train = model.train(data["x_train"][0], w_init, b_init)
  cost_init = model.cost(data["x_train"], data["y_train"], w_init, b_init)
  w_train, b_train = model.gradient_descent(data["x_train"], data["y_train"], w_init, b_init, 0.001, 100000)

  # validate model
  y_predict_val = model.train(data["x_val"][0], w_train, b_train)
  cost_val = model.cost(data["x_val"], data["y_val"], w_train, b_train)

  # test model
  y_predict_test = model.train(data["x_test"][10], w_train, b_train)
  cost_test = model.cost(data["x_test"], data["y_test"], w_train, b_train)

  print("cost init: ", cost_init)
  print("cost val: ", cost_val)
  print("cost test: ", cost_test)

  print("=" * 100)
  print("w train: ", w_train)
  print("b train: ", b_train)

  print("=" * 100)
  print("y val: ", np.expm1(data["y_val"][0]), f"({data['y_val'][0]})")
  print("y predict val: ", np.expm1(y_predict_val), f"({y_predict_val})")

  print("=" * 100)
  print("y test: ", np.expm1(data["y_test"][10]), f"({data['y_test'][10]})")
  print("y predict test: ", np.expm1(y_predict_test), f"({y_predict_test})")

  