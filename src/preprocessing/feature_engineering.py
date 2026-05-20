from src.database.load_data import LoadData


class FeatureEngineering:

    @staticmethod
    def build_dataset():

        df = LoadData.load_meteorological_data()

        df["alagamento"] = (df["rain_24h"] >= 80).astype(int)

        FEATURES = [
            "precipitation",
            "temperature",
            "humidity",
            "rain_3h",
            "rain_6h",
            "rain_12h",
            "rain_24h",
            "hour_of_day",
            "month",
            "day_of_week",
        ]

        X = df[FEATURES]
        y = df["alagamento"]
        
        return X, y
    
    
