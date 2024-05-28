import numpy as np
import pickle

# Load the trained model
with open("studentgrades.pickle", "rb") as f:
    model = pickle.load(f)

# New student data (example: [G1, G2, absences, failures, studytime])
new_students_data = np.array([
    [15, 14, 3, 2, 4],
    [10, 9, 0, 0, 2],
    [12, 13, 5, 1, 3]
])

# Make a prediction
predicted_grades = model.predict(new_students_data)
print(f"Predicted Grades: {predicted_grades}")
