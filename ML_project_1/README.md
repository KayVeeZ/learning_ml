# Dragon Real Estate Price Predictor

Welcome to the Dragon Real Estate Price Predictor! This project aims to predict real estate prices using various machine learning models. The data and analysis are done using Python, with a focus on understanding the data, preprocessing, and model evaluation.

## Table of Contents

- [Installation](#installation)
- [Project Overview](#project-overview)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

To get started with this project, you'll need to have Python installed. Follow these steps to set up your environment:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/dragon-real-estate.git
    cd dragon-real-estate
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4. Install the required packages:
    ```bash
    pip install numpy pandas matplotlib scikit-learn
    ```

## Project Overview

This project involves several key steps:
1. **Data Exploration**: Understanding the structure and distribution of the data.
2. **Data Preprocessing**: Handling missing values, creating new attributes, and scaling features.
3. **Model Training**: Training various machine learning models to predict real estate prices.
4. **Model Evaluation**: Evaluating model performance using cross-validation and other metrics.

## Data Exploration

The data exploration process involves:
- Loading the dataset.
- Viewing basic information about the dataset.
- Visualizing data distributions and relationships.
- Looking for correlations between features.

## Data Preprocessing

Data preprocessing steps include:
- Splitting the data into training and test sets.
- Handling missing values using strategies like imputation.
- Creating new attributes to enhance model performance.
- Scaling features to ensure consistent data ranges.

## Model Training and Evaluation

Different models are trained and evaluated in this project:
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**

The models are evaluated using:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Cross-validation scores

## Results

The results of the model evaluations are printed and compared to select the best-performing model. The Random Forest Regressor showed the most promising results in terms of accuracy and stability.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to submit a pull request. For major changes, please open an issue first to discuss what you would like to change.
