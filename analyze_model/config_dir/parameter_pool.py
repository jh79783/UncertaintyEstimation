class LossComb:
    STANDARD = {"mse": ([1.], "MSELoss")}


class TrainingPlan:
    ANAYLZE = [("analyze", 10, 0.001, LossComb.STANDARD)]


class TfrParams:
    MIN_PIX = {'train': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0},
               'val': {"Bgd": 0, "Pedestrian": 0, "Car": 0, "Cyclist": 0
                       }
               }

    CATEGORY_NAMES = {"category": ["Bgd", "Pedestrian", "Car", "Cyclist"],
                      "dont": ["DontCare"]
                      }

