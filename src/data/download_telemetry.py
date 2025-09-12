import kagglehub
import os
import shutil

def download_telemetry_dataset():

    print("Скачивание датасета 'yunlevin/levin-vehicle-telematics'...")
    try:
        path = kagglehub.dataset_download("yunlevin/levin-vehicle-telematics")
        print(f"Датасет скачан в: {path}")

        possible_files = {
            'obd': ['allcars.csv', 'v2.csv', 'obd_data.csv', 'OBD_Data.csv'],
            'trip': ['tripdata.csv', 'trip_data.csv']
        }

        raw_dir = "data/raw"
        os.makedirs(raw_dir, exist_ok=True)

        found_obd = False
        found_trip = False

        for file in os.listdir(path):
            src = os.path.join(path, file)

            if not found_obd and any(name.lower() in file.lower() for name in possible_files['obd']):
                dst = os.path.join(raw_dir, 'obd_data.csv')
                print(f"Найден OBD-файл: {file} → копируем как obd_data.csv")
                shutil.copy(src, dst)
                found_obd = True

            if not found_trip and any(name.lower() in file.lower() for name in possible_files['trip']):
                dst = os.path.join(raw_dir, 'tripdata.csv')
                print(f"Найден Trip-файл: {file} → копируем как tripdata.csv")
                shutil.copy(src, dst)
                found_trip = True

        if found_obd:
            print("OBD-данные готовы к обработке.")
        else:
            print("Не найден подходящий OBD-файл для обработки.")

        if found_trip:
            print("Trip-данные готовы.")

    except Exception as e:
        print(f"Ошибка при работе с датасетом: {e}")

if __name__ == "__main__":
    download_telemetry_dataset()