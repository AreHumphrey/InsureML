from src.models.lightgbm.telematics_model import TelematicsRiskModel
import pandas as pd

def main():

    df = pd.read_csv("src/data/raw/telematics_data.csv")
    X = df.drop(columns=["target", "TripID"], errors='ignore')
    y = df["target"]

    model = TelematicsRiskModel()
    model.train(X, y)

    model.save_model("outputs/lightgbm/telematics_model_v1.pkl")
    print("Модель LightGBM обучена и сохранена!")

if __name__ == "__main__":
    main()