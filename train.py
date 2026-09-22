import os
import cv2
import numpy as np

# ==============================
# SETTINGS
# ==============================

DATASET_PATH = "dataset/faces"

IMAGE_SIZE = 100

K_VALUES = [10, 20, 30, 40, 50, 60, 80]

HIDDEN_SIZE = 50
LEARNING_RATE = 0.01
EPOCHS = 3000

np.random.seed(42)


# ==============================
# LOAD DATASET
# ==============================

def load_dataset():

    faces = []
    labels = []
    names = []

    if not os.path.exists(DATASET_PATH):

        print("Dataset path not found!")
        print("Expected path:")
        print(os.path.abspath(DATASET_PATH))

        return None, None, None

    folders = sorted(os.listdir(DATASET_PATH))

    for folder in folders:

        folder_path = os.path.join(
            DATASET_PATH,
            folder
        )

        if not os.path.isdir(folder_path):
            continue

        label = len(names)

        names.append(folder)

        for file in sorted(os.listdir(folder_path)):

            file_path = os.path.join(
                folder_path,
                file
            )

            image = cv2.imread(
                file_path,
                cv2.IMREAD_GRAYSCALE
            )

            if image is None:
                continue

            # Resize image
            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            # Convert to float
            image = image.astype(
                np.float64
            )

            # Normalize
            image = image / 255.0

            # Convert image to vector
            image = image.flatten()

            faces.append(image)

            labels.append(label)

    faces = np.array(faces)
    labels = np.array(labels)

    print("--------------------------------")
    print("Number of images:", len(faces))
    print("Image matrix shape:", faces.shape)
    print("Names:", names)
    print("--------------------------------")

    return faces, labels, names


# ==============================
# TRAIN TEST SPLIT
# 80% TRAIN
# 20% TEST
# ==============================

def split_data(X, y):

    train_indices = []
    test_indices = []

    classes = np.unique(y)

    for c in classes:

        indices = np.where(
            y == c
        )[0]

        np.random.shuffle(indices)

        # 80 percent training
        split = int(
            0.80 * len(indices)
        )

        train_indices.extend(
            indices[:split]
        )

        test_indices.extend(
            indices[split:]
        )

    train_indices = np.array(
        train_indices
    )

    test_indices = np.array(
        test_indices
    )

    return (
        X[train_indices],
        X[test_indices],
        y[train_indices],
        y[test_indices]
    )


# ==============================
# PCA
# ==============================

def perform_pca(X_train, k):

    print("\nCalculating Mean Face...")

    # Calculate mean face
    mean_face = np.mean(
        X_train,
        axis=0
    )

    # Mean center data
    A = X_train - mean_face

    print(
        "Mean zero matrix shape:",
        A.shape
    )

    # Covariance surrogate matrix
    covariance = np.dot(
        A,
        A.T
    )

    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )

    # Sort in descending order
    order = np.argsort(
        eigenvalues
    )[::-1]

    eigenvectors = eigenvectors[
        :,
        order
    ]

    # Make sure k is valid
    k = min(
        k,
        eigenvectors.shape[1]
    )

    # Select top k eigenvectors
    selected_vectors = eigenvectors[
        :,
        :k
    ]

    # Generate eigenfaces
    eigenfaces = np.dot(
        selected_vectors.T,
        A
    )

    # Normalize eigenfaces
    for i in range(
        len(eigenfaces)
    ):

        norm = np.linalg.norm(
            eigenfaces[i]
        )

        if norm != 0:

            eigenfaces[i] = (
                eigenfaces[i] / norm
            )

    print(
        "Selected eigenfaces shape:",
        eigenfaces.shape
    )

    # Project training images
    signatures = np.dot(
        A,
        eigenfaces.T
    )

    return (
        mean_face,
        eigenfaces,
        signatures
    )


# ==============================
# SOFTMAX
# ==============================

def softmax(z):

    z = z - np.max(
        z,
        axis=1,
        keepdims=True
    )

    exp_z = np.exp(z)

    return exp_z / np.sum(
        exp_z,
        axis=1,
        keepdims=True
    )


# ==============================
# TRAIN ANN
# ==============================

def train_ann(
    X,
    y,
    number_of_classes
):

    input_size = X.shape[1]

    # Initialize weights
    W1 = np.random.randn(
        input_size,
        HIDDEN_SIZE
    ) * np.sqrt(
        1 / input_size
    )

    b1 = np.zeros(
        (1, HIDDEN_SIZE)
    )

    W2 = np.random.randn(
        HIDDEN_SIZE,
        number_of_classes
    ) * np.sqrt(
        1 / HIDDEN_SIZE
    )

    b2 = np.zeros(
        (1, number_of_classes)
    )

    # ==========================
    # ONE HOT ENCODING
    # ==========================

    Y = np.zeros(
        (
            len(y),
            number_of_classes
        )
    )

    Y[
        np.arange(len(y)),
        y
    ] = 1


    # ==========================
    # TRAINING LOOP
    # ==========================

    for epoch in range(EPOCHS):

        # ----------------------
        # FORWARD PROPAGATION
        # ----------------------

        Z1 = np.dot(
            X,
            W1
        ) + b1

        A1 = np.tanh(
            Z1
        )

        Z2 = np.dot(
            A1,
            W2
        ) + b2

        A2 = softmax(
            Z2
        )


        # ----------------------
        # BACK PROPAGATION
        # ----------------------

        m = len(X)

        dZ2 = (
            A2 - Y
        ) / m

        dW2 = np.dot(
            A1.T,
            dZ2
        )

        db2 = np.sum(
            dZ2,
            axis=0,
            keepdims=True
        )

        dA1 = np.dot(
            dZ2,
            W2.T
        )

        dZ1 = dA1 * (
            1 - A1 ** 2
        )

        dW1 = np.dot(
            X.T,
            dZ1
        )

        db1 = np.sum(
            dZ1,
            axis=0,
            keepdims=True
        )


        # ----------------------
        # UPDATE WEIGHTS
        # ----------------------

        W1 -= (
            LEARNING_RATE * dW1
        )

        b1 -= (
            LEARNING_RATE * db1
        )

        W2 -= (
            LEARNING_RATE * dW2
        )

        b2 -= (
            LEARNING_RATE * db2
        )


        # Show training accuracy
        if epoch % 500 == 0:

            predictions = np.argmax(
                A2,
                axis=1
            )

            accuracy = np.mean(
                predictions == y
            ) * 100

            print(
                "Epoch:",
                epoch,
                "| Training Accuracy:",
                round(
                    accuracy,
                    2
                ),
                "%"
            )

    return (
        W1,
        b1,
        W2,
        b2
    )


# ==============================
# PREDICT
# ==============================

def predict_ann(
    X,
    W1,
    b1,
    W2,
    b2
):

    Z1 = np.dot(
        X,
        W1
    ) + b1

    A1 = np.tanh(
        Z1
    )

    Z2 = np.dot(
        A1,
        W2
    ) + b2

    probabilities = softmax(
        Z2
    )

    predictions = np.argmax(
        probabilities,
        axis=1
    )

    return predictions


# ==============================
# MAIN
# ==============================

faces, labels, names = load_dataset()

if faces is None:

    exit()


# ==============================
# SPLIT DATA
# ==============================

X_train, X_test, y_train, y_test = split_data(
    faces,
    labels
)

print(
    "\nTraining data shape:",
    X_train.shape
)

print(
    "Testing data shape:",
    X_test.shape
)


# ==============================
# VARIABLES FOR BEST MODEL
# ==============================

accuracy_results = []

best_accuracy = -1

best_k = None

best_model = None


# ==============================
# TEST DIFFERENT K VALUES
# ==============================

for k in K_VALUES:

    print(
        "\n================================"
    )

    print(
        "PCA with k =",
        k
    )

    print(
        "================================"
    )


    # PCA
    mean_face, eigenfaces, train_signatures = perform_pca(
        X_train,
        k
    )


    # ==========================
    # TEST SIGNATURES
    # ==========================

    test_zero = (
        X_test - mean_face
    )

    test_signatures = np.dot(
        test_zero,
        eigenfaces.T
    )


    print(
        "Training ANN..."
    )


    # ==========================
    # TRAIN ANN
    # ==========================

    W1, b1, W2, b2 = train_ann(
        train_signatures,
        y_train,
        len(names)
    )


    # ==========================
    # TEST PREDICTION
    # ==========================

    predictions = predict_ann(
        test_signatures,
        W1,
        b1,
        W2,
        b2
    )


    correct = np.sum(
        predictions == y_test
    )

    accuracy = (
        correct / len(y_test)
    ) * 100


    accuracy_results.append(
        (
            k,
            accuracy
        )
    )


    print(
        "\nCorrect predictions:",
        correct
    )

    print(
        "Total testing images:",
        len(y_test)
    )

    print(
        "Test Accuracy:",
        round(
            accuracy,
            2
        ),
        "%"
    )


    # ==========================
    # SAVE BEST MODEL
    # ==========================

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_k = k

        best_model = (
            mean_face.copy(),
            eigenfaces.copy(),
            W1.copy(),
            b1.copy(),
            W2.copy(),
            b2.copy()
        )


# ==============================
# SAVE BEST MODEL
# ==============================

(
    mean_face,
    eigenfaces,
    W1,
    b1,
    W2,
    b2
) = best_model


np.savez(
    "model.npz",

    mean_face=mean_face,

    eigenfaces=eigenfaces,

    W1=W1,

    b1=b1,

    W2=W2,

    b2=b2,

    names=np.array(names)
)


# ==============================
# FINAL RESULT
# ==============================

print(
    "\n================================"
)

print(
    "FINAL RESULT"
)

print(
    "================================"
)

print(
    "Best k:",
    best_k
)

print(
    "Best Accuracy:",
    round(
        best_accuracy,
        2
    ),
    "%"
)

print(
    "\nModel saved successfully!"
)

print(
    "File: model.npz"
)


# ==============================
# SAVE ACCURACY RESULTS
# ==============================

with open(
    "accuracy_results.txt",
    "w"
) as file:

    file.write(
        "PCA + ANN Accuracy Results\n"
    )

    file.write(
        "==========================\n\n"
    )

    for k, accuracy in accuracy_results:

        file.write(
            "k = "
            + str(k)
            + " | Accuracy = "
            + str(
                round(
                    accuracy,
                    2
                )
            )
            + "%\n"
        )


    file.write(
        "\nBest k = "
        + str(best_k)
    )

    file.write(
        "\nBest Accuracy = "
        + str(
            round(
                best_accuracy,
                2
            )
        )
        + "%"
    )


print(
    "Accuracy results saved!"
)

print(
    "\nTraining completed successfully!"
)