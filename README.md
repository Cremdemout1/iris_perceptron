# Iris Perceptron Predictor

This project is a simple AI that uses a Perceptron model to classify Iris flowers based on their sepal length and petal length. It uses the Iris dataset from the UCI Machine Learning Repository.

## Requirements

To run this project, you'll need to have the following installed on your local machine:

1. **Python 3.x**  
   Ensure you have Python 3.6 or higher installed. You can download it from [here](https://www.python.org/downloads/).

2. **Pip**  
   You can install Python packages using pip. It comes pre-installed with Python, but if you don't have it, you can follow the instructions [here](https://pip.pypa.io/en/stable/installation/).

## Setting Up the Environment

### 1. Clone the Repository

If you haven't cloned the repository yet, do so by running:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Create a Virtual Environment (Optional, but Recommended)

Creating a virtual environment ensures you don't interfere with your global Python environment. To create one, run the following command:

```bash
python3 -m venv iris_predictor_env
```

Activate the virtual environment:

- **On Windows**:
  ```bash
  iris_predictor_env\Scripts\activate
  ```

- **On macOS/Linux**:
  ```bash
  source iris_predictor_env/bin/activate
  ```

### 3. Install the Required Libraries

Create a `requirements.txt` file in your project folder with the following content:

```txt
pandas
numpy
requests
matplotlib
```

Now, install the necessary libraries:

```bash
pip install -r requirements.txt
```

Alternatively, you can manually install the libraries using the following commands:

```bash
pip install pandas numpy requests matplotlib
```

### 4. Run the Script

Once your environment is set up, you can run the script:

```bash
python iris_predictor.py
```

This will load the Iris dataset, train the Perceptron model, and prompt you for user input to predict whether an Iris flower is of type **Setosa** or **Versicolor** based on its sepal and petal length.

### 5. Exit or Predict

- Type `exit` to leave the program.
- Type `examine` to enter the Iris predictor mode, where you will be asked to input the sepal and petal length for the prediction.

## Example Interaction

```bash
Please write 'exit' to leave and 'examine' to enter AI Iris predictor
examine
Enter sepal length of Iris: 5.1
Enter petal length of Iris: 1.4
Iris Setosa
```

## How It Works

This project uses a **Perceptron** model, which is a simple binary classifier in machine learning. It works by learning a linear decision boundary between two classes based on input data. Here's how it works:

1. **Dataset**: We use the Iris dataset, which contains various features like sepal length, petal length, sepal width, and petal width for different species of Iris flowers. For this project, we focus on the first 100 samples of the dataset, which belong to two species: **Iris-setosa** and **Iris-versicolor**.

2. **Preprocessing**: We select two features, **sepal length** and **petal length**, to predict whether a flower is **Iris-setosa** (represented by `-1`) or **Iris-versicolor** (represented by `1`).

3. **Training**: The Perceptron is trained using these two features for each flower. During training, it adjusts the weights of the model iteratively by making small corrections based on the prediction errors. The learning process continues for a fixed number of iterations.

4. **Prediction**: Once trained, the Perceptron model can classify new samples. The user is prompted to input the sepal and petal lengths of an Iris flower. Based on the trained model, it will predict whether the flower is **Iris-setosa** or **Iris-versicolor**.

5. **Output**: The user is given a classification result: either **Iris Setosa** or **Iris Versicolor**, based on the model's prediction.
