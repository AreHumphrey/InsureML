import pandas as pd
import os


def load_dataset(file_path: str) -> pd.DataFrame:

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл не найден: {file_path}")

    df = pd.read_csv(file_path)
    print(f"Датасет загружен: {len(df)} строк, {len(df.columns)} столбцов")
    return df