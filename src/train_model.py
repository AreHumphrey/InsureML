from src.models.insurance_model import InsuranceRiskModel
from src.data.raw.load_data import load_dataset
from src.data.raw.preprocess_data import preprocess


def main():

    df = load_dataset("data/raw/insurance_data.csv")
    df_clean = preprocess(df)

    X = df_clean.drop(columns=["target"])
    y = df_clean["target"]

    model = InsuranceRiskModel()
    model.train(X, y)

    model.save_model("outputs/insurance_model_v1.pkl")
    print("Модель успешно обучена и сохранена.")


if __name__ == "__main__":
    main()