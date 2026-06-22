import pandas as pd


def check_dtc_in_file(file_path: str) -> bool:

    try:
        df = pd.read_csv(file_path, low_memory=False)

        if 'dtc' not in df.columns:
            print("Колонка dtc не найдена в файле.")
            return False

        dtc_vals = pd.to_numeric(df['dtc'], errors='coerce').fillna(0)
        has_dtc_errors = (dtc_vals > 0).any()

        print(f"Проверено {len(df)} строк. Найдены DTC: {has_dtc_errors}")
        return has_dtc_errors

    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return False