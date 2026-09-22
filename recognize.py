import cv2
import numpy as np
import os

# ==========================================
# SETTINGS
# ==========================================

IMAGE_SIZE = 100

# Confidence below this = Unknown / Impostor
UNKNOWN_THRESHOLD = 0.50


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

if not os.path.exists("model.npz"):
    print("ERROR: model.npz not found!")
    print("Please run train.py first.")
    exit()

model = np.load(
    "model.npz",
    allow_pickle=True
)

mean_face = model["mean_face"]
eigenfaces = model["eigenfaces"]

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

names = np.asarray(
    model["names"]
).flatten()


# ==========================================
# SOFTMAX
# ==========================================

def softmax(z):

    z = np.asarray(z).flatten()

    z = z - np.max(z)

    exp_z = np.exp(z)

    return exp_z / np.sum(exp_z)


# ==========================================
# FACE RECOGNITION
# ==========================================

def recognize(image_path):

    # Remove quotes if user pasted them
    image_path = image_path.strip().strip('"')

    # Check file
    if not os.path.exists(image_path):

        print("--------------------------------")
        print("ERROR: Image file not found!")
        print("--------------------------------")
        print("Path used:")
        print(image_path)
        return

    # Read image
    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:

        print("--------------------------------")
        print("ERROR: Could not read image!")
        print("--------------------------------")
        return

    print("Image loaded successfully!")

    # ======================================
    # RESIZE
    # ======================================

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # ======================================
    # CONVERT IMAGE INTO VECTOR
    # ======================================

    image = image.astype(
        np.float64
    )

    image = image / 255.0

    image = image.flatten()

    # ======================================
    # SUBTRACT MEAN FACE
    # ======================================

    test_zero = (
        image - mean_face
    )

    # ======================================
    # PCA PROJECTION
    # ======================================

    signature = np.dot(
        test_zero,
        eigenfaces.T
    )

    signature = np.asarray(
        signature
    ).flatten()

    # ======================================
    # ANN FORWARD PROPAGATION
    # ======================================

    Z1 = np.dot(
        signature,
        W1
    ) + b1.flatten()

    A1 = np.tanh(Z1)

    Z2 = np.dot(
        A1,
        W2
    ) + b2.flatten()

    # ======================================
    # CALCULATE PROBABILITIES
    # ======================================

    probabilities = softmax(
        Z2
    )

    # Highest probability
    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    # ======================================
    # DISPLAY RESULT
    # ======================================

    print("--------------------------------")
    print("Face Recognition Result")
    print("--------------------------------")

    print(
        "Highest confidence:",
        round(
            confidence * 100,
            2
        ),
        "%"
    )

    print(
        "Threshold:",
        UNKNOWN_THRESHOLD * 100,
        "%"
    )

    # ======================================
    # UNKNOWN / IMPOSTOR CHECK
    # ======================================

    if confidence < UNKNOWN_THRESHOLD:

        print(
            "Recognized person:",
            "Unknown / Impostor"
        )

    else:

        print(
            "Recognized person:",
            names[predicted_index]
        )

    print("--------------------------------")


# ==========================================
# TAKE IMAGE PATH FROM USER
# ==========================================

image_path = input(
    "Enter test image path: "
)

recognize(image_path)