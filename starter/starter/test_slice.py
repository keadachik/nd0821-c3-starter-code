# テスト用の小さなデータ
import pandas as pd
import numpy as np
from ml.model import compute_slice_metrics

test_df = pd.DataFrame({
    'education': ['Bachelors', 'Bachelors', 'Masters', 'HS-grad'],
    'age': [25, 30, 35, 40]
})
test_y_true = np.array([1, 0, 1, 0])
test_y_pred = np.array([1, 0, 0, 1])

# 関数を実行
result = compute_slice_metrics(test_df, 'education', test_y_true, test_y_pred)
print(result)
