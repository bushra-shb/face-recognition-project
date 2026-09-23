# Face Recognition Project

A Python-based face recognition system using NumPy, SciPy, OpenCV and an Artificial Neural Network.

## Technologies Used

- Python
- NumPy
- SciPy
- OpenCV
- Artificial Neural Network

## Project Files

- train.py - Trains the face recognition model
- recognize.py - Recognizes faces from test images
- graph.py - Generates the accuracy graph
- accuracy_graph.jpg - Accuracy graph
- accuracy_results.txt - Accuracy results
- model.npz - Trained model
- requirements.txt - Required Python libraries

## How the Project Works

1. Face images are loaded from the dataset.
2. Images are converted into numerical data.
3. Face features are extracted.
4. The Artificial Neural Network is trained using these features.
5. A test image is given to the recognition system.
6. The trained model predicts the corresponding person.

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
python train.py
python recognize.py
## Results
The project includes accuracy results and an accuracy graph generated during testing.
