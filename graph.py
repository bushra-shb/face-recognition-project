import cv2
import numpy as np

# PCA k values
k_values = [5, 10, 20, 30, 40, 50]

# Accuracy obtained during testing
accuracies = [35.00, 43.33, 49.44, 55.56, 54.44, 56.67]

# Graph settings
width = 900
height = 600

img = np.ones((height, width, 3), dtype=np.uint8) * 255

# Graph area
left = 100
right = 800
top = 80
bottom = 500

# Draw axes
cv2.line(img, (left, bottom), (right, bottom), (0, 0, 0), 2)
cv2.line(img, (left, top), (left, bottom), (0, 0, 0), 2)

# Title
cv2.putText(
    img,
    "PCA k Value vs Accuracy",
    (250, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

# Y-axis labels: 0 to 100
for value in range(0, 101, 20):
    y = bottom - int((value / 100) * (bottom - top))

    cv2.line(img, (left - 5, y), (right, y), (220, 220, 220), 1)

    cv2.putText(
        img,
        str(value),
        (55, y + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

# Plot points
points = []

for k, accuracy in zip(k_values, accuracies):

    x = left + int(((k - 5) / (50 - 5)) * (right - left))
    y = bottom - int((accuracy / 100) * (bottom - top))

    points.append((x, y))

    cv2.circle(img, (x, y), 6, (0, 0, 255), -1)

    cv2.putText(
        img,
        str(accuracy) + "%",
        (x - 25, y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    cv2.putText(
        img,
        "k=" + str(k),
        (x - 15, bottom + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

# Join points
for i in range(len(points) - 1):
    cv2.line(
        img,
        points[i],
        points[i + 1],
        (0, 0, 255),
        2
    )

# Axis names
cv2.putText(
    img,
    "PCA k Value",
    (380, 560),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 0, 0),
    2
)

cv2.putText(
    img,
    "Accuracy (%)",
    (10, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 0, 0),
    2
)

# Save graph
cv2.imwrite("accuracy_graph.jpg", img)

print("--------------------------------")
print("Graph created successfully!")
print("File: accuracy_graph.jpg")
print("--------------------------------")