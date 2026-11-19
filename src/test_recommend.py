from recommend import get_recommendations

base = {"male": 0, "age": 50, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 0,
        "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, "totChol": 180,
        "sysBP": 115, "diaBP": 75, "BMI": 24, "heartRate": 70, "glucose": 90}

cases = [
    ("A normal", base, ["R11"]),
    ("B stage 1 + borderline cholesterol",
     {**base, "sysBP": 135, "diaBP": 85, "totChol": 210}, ["R2", "R5", "R11"]),
    ("C crisis BP + high cholesterol + BP meds",
     {**base, "sysBP": 190, "diaBP": 125, "totChol": 250, "BPMeds": 1}, ["R4", "R6", "R10", "R11"]),
    ("D smoker, BMI 31, glucose 130",
     {**base, "cigsPerDay": 20, "currentSmoker": 1, "BMI": 31, "glucose": 130},
     ["R7", "R8b", "R9b", "R11"]),
    ("E BMI 27, glucose 105", {**base, "BMI": 27, "glucose": 105}, ["R8a", "R9a", "R11"]),
    ("F lower boundaries: BMI 25, glucose 100", {**base, "BMI": 25, "glucose": 100},
     ["R8a", "R9a", "R11"]),
    ("G upper boundaries: BMI 30, glucose 126", {**base, "BMI": 30, "glucose": 126},
     ["R8b", "R9b", "R11"]),
]

for name, patient, expected in cases:
    got = [r["id"] for r in get_recommendations(patient)]
    print("OK  " if got == expected else "FAIL", name, "->", got)
    assert got == expected, f"expected {expected}"

print("\nAll", len(cases), "tests passed")