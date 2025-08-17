import pandas as pd
import numpy as np

class LogisticRegression:
  def __init__(self):
    pass

  def load(self, path: str) -> pd.DataFrame:
    csv =  pd.read_csv(path)
    return pd.DataFrame(csv)

  def prepare(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df.columns = df.columns.str.lower()

    normalize_col = ["area", "majoraxislength", "minoraxislength", "convexarea", "perimeter"]
    for col in normalize_col:
      df[col] = self.min_max_normalize(df[col])
    return df

  def min_max_normalize(self, x: np.ndarray) -> np.ndarray:
    return (x - x.min()) / (x.max() - x.min())

  def split_dataset(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = len(df)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_df = df.iloc[:int(m * 0.6)].copy()
    val_df = df.iloc[int(m * 0.6):int(m * 0.8)].copy()
    test_df = df.iloc[int(m * 0.8):].copy()

    train_y = np.array(train_df["class"].map({"Kecimen": 0, "Besni": 1}))
    val_y = np.array(val_df["class"].map({"Kecimen": 0, "Besni": 1}))
    test_y = np.array(test_df["class"].map({"Kecimen": 0, "Besni": 1}))

    del train_df["class"]
    del val_df["class"]
    del test_df["class"]

    train_x = np.array(train_df)
    val_x = np.array(val_df)
    test_x = np.array(test_df)

    return {
      "train_x": train_x,
      "val_x": val_x,
      "test_x": test_x,
      "train_y": train_y,
      "val_y": val_y,
      "test_y": test_y
    }

  def train(self, x: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return self.sigmoid(np.dot(x, w) + b)

  def sigmoid(self, z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))

  def cost(self, x: np.ndarray, y: np.ndarray, w: np.ndarray, b: float) -> float:
    m = len(y)
    cost = 0

    for i in range(m):
      f_x = self.train(x[i], w, b)
      cost += (-(np.dot(y[i], np.log(f_x))) - np.dot((1 - y[i]), np.log(1 - f_x)))

    return cost / m

  def gradient_descent(self, x, y, w, b, alpha, epochs):
    w_init = w
    b_init = b
    
    for i in range(epochs):
      dw, db = self.derivative(w_init, b_init, x, y)
      w_init = w_init - alpha * dw
      b_init = b_init - alpha * db

      if i % (epochs / 10) == 0:
        print(f"Epoch {i} - Cost: {self.cost(x, y, w_init, b_init)}")

    return w_init, b_init

  def derivative(self, w, b, x, y):
    m = x.shape[0]
    dw = 0
    db = 0

    for i in range(m):
      f_x = self.train(x[i], w, b)
      dw += (f_x - y[i]) * x[i]
      db += (f_x - y[i])

    return dw / m, db / m


if __name__ == "__main__":
  model = LogisticRegression()
  df = model.load("data/Raisin_Dataset.csv")
  prepared_df = model.prepare(df)
  dataset = model.split_dataset(prepared_df)

  w_init = np.zeros(dataset["train_x"].shape[1])
  b_init = 10

  y_predict_init = model.train(dataset["train_x"][0], w_init, b_init)
  cost_init = model.cost(dataset["train_x"], dataset["train_y"], w_init, b_init)
  w_init, b_init = model.gradient_descent(dataset["train_x"], dataset["train_y"], w_init, b_init, 0.01, 100000)

  print(w_init, b_init)

  y_predict_val = model.train(dataset["val_x"][0], w_init, b_init)
  cost_val = model.cost(dataset["train_x"], dataset["train_y"], w_init, b_init)

  print(y_predict_val, dataset["val_y"][0])
  print("cost_val", cost_val)

  y_predict_test = model.train(dataset["test_x"][10], w_init, b_init)
  cost_test = model.cost(dataset["test_x"], dataset["test_y"], w_init, b_init)

  print(y_predict_test, dataset["test_y"][10])
  print("cost_test", cost_test)
