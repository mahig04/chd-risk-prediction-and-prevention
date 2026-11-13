DISCLAIMER = ("Educational prototype only. This is not a diagnosis or medical advice. "
              "Please consult a qualified healthcare professional.")

BP_NOTE = "Based on a single reading; guidelines use the average of two or more readings."


def bp_category(sys_bp, dia_bp):
    # if systolic and diastolic fall in different categories, the higher one applies
    if sys_bp > 180 or dia_bp > 120:
        return "crisis"
    if sys_bp >= 140 or dia_bp >= 90:
        return "stage2"
    if sys_bp >= 130 or dia_bp >= 80:
        return "stage1"
    if sys_bp >= 120:
        return "elevated"
    return "normal"


# Each rule: id, condition on the patient dict, advice text, source
RULES = [
    ("R1", lambda p: bp_category(p["sysBP"], p["diaBP"]) == "elevated",
     "Your blood pressure is in the 'elevated' range. A healthy diet, less salt, "
     "regular activity and regular BP checks are recommended. " + BP_NOTE,
     "ACC/AHA blood pressure guideline"),
    ("R2", lambda p: bp_category(p["sysBP"], p["diaBP"]) == "stage1",
     "Your blood pressure is in the stage 1 range. Lifestyle changes are the first step. "
     "Recheck in 3-6 months and discuss it with a doctor. " + BP_NOTE,
     "ACC/AHA blood pressure guideline"),
    ("R3", lambda p: bp_category(p["sysBP"], p["diaBP"]) == "stage2",
     "Your blood pressure is in the stage 2 range. Please see a doctor within about a month. " + BP_NOTE,
     "ACC/AHA blood pressure guideline"),
    ("R4", lambda p: bp_category(p["sysBP"], p["diaBP"]) == "crisis",
     "This blood pressure reading is very high. Please contact a healthcare professional promptly. " + BP_NOTE,
     "AHA hypertensive crisis range (advice wording to be confirmed)"),
    ("R5", lambda p: 200 <= p["totChol"] < 240,
     "Your total cholesterol is borderline high. Focus on diet and activity, and ask your doctor "
     "for a full lipid profile.",
     "NHLBI/NCEP total cholesterol categories"),
    ("R6", lambda p: p["totChol"] >= 240,
     "Your total cholesterol is high. Please see a doctor for a full lipid profile.",
     "NHLBI/NCEP total cholesterol categories"),
    ("R10", lambda p: p["prevalentHyp"] == 1 or p["BPMeds"] == 1,
     "Keep following your doctor's treatment plan and monitor your blood pressure regularly, "
     "including at home if advised.",
     "ACC/AHA blood pressure guideline"),
    ("R11", lambda p: True,
     "Aim for 150-300 minutes of moderate physical activity per week, plus muscle-strengthening "
     "activity on 2 or more days.",
     "WHO 2020 physical activity guidelines"),
]


def get_recommendations(patient):
    out = []
    for rule_id, condition, advice, source in RULES:
        if condition(patient):
            out.append({"id": rule_id, "advice": advice, "source": source})
    return out


if __name__ == "__main__":
    base = {"male": 0, "age": 50, "currentSmoker": 0, "cigsPerDay": 0, "BPMeds": 0,
            "prevalentStroke": 0, "prevalentHyp": 0, "diabetes": 0, "totChol": 180,
            "sysBP": 115, "diaBP": 75, "BMI": 24, "heartRate": 70, "glucose": 90}
    tests = {
        "A: normal": base,
        "B: stage 1 + borderline cholesterol": {**base, "sysBP": 135, "diaBP": 85, "totChol": 210},
        "C: very high BP + high cholesterol + on BP meds": {**base, "sysBP": 190, "diaBP": 125,
                                                          "totChol": 250, "BPMeds": 1},
    }
    for name, patient in tests.items():
        ids = [r["id"] for r in get_recommendations(patient)]
        print(name, "->", ids)
    print("\n" + DISCLAIMER)