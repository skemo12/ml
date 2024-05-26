import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.calibration import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.feature_selection import (SelectPercentile, VarianceThreshold,
                                       f_classif)
from sklearn.impute import IterativeImputer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer, RobustScaler


def prelucrate_data(df):
    df["Sedentary_hours_daily"] = (
        df["Sedentary_hours_daily"].str.replace(",", ".").astype(float)
    )
    df["Age"] = df["Age"].str.replace(",", ".").astype(float).astype(int)
    df["Est_avg_calorie_intake"] = df["Est_avg_calorie_intake"].astype(int)
    df["Height"] = df["Height"].str.replace(",", ".").astype(float)
    df["Water_daily"] = df["Water_daily"].str.replace(",", ".").astype(float)
    df["Weight"] = df["Weight"].str.replace(",", ".").astype(float)
    df["Physical_activity_level"] = (
        df["Physical_activity_level"].str.replace(",", ".").astype(float)
    )
    df["Technology_time_use"] = df["Technology_time_use"].astype(object)
    df["Main_meals_daily"] = (
        df["Main_meals_daily"]
        .str.replace(",", ".")
        .astype(float)
        .astype(int)
        .astype(object)
    )
    df["Regular_fiber_diet"] = (
        df["Regular_fiber_diet"].str.replace(",", ".").astype(float).astype(int)
    )


def prepare_dataset():
    df = pd.read_csv("date_tema_1_iaut_2024.csv")
    prelucrate_data(df)
    # Replace -1 with NaN in the 'Weight' column
    df["Weight"] = df["Weight"].replace(-1, np.nan)
    # df['Age'] = df['Age'].mask(df['Age'] >= 100)
    # Initialize the IterativeImputer
    imputer = IterativeImputer()

    # Perform the imputation on the 'Weight' column
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df = df.replace(-1, np.nan)
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    # Convert categorical columns to numerical
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])

    X = df.drop("Diagnostic", axis=1)
    y = df["Diagnostic"]

    # Create a VarianceThreshold object
    selector = VarianceThreshold(threshold=0.1)

    # Fit and transform the selector to the data
    features_before = X.columns
    X = pd.DataFrame(
        selector.fit_transform(X), columns=X.columns[selector.get_support()]
    )
    print(f"Features removed: {set(features_before) - set(X.columns)}")

    # Create a SelectPercentile object
    selector = SelectPercentile(f_classif, percentile=90)

    # Fit and transform the selector to the data
    features_before = X.columns
    X = pd.DataFrame(
        selector.fit_transform(X, y), columns=X.columns[selector.get_support()]
    )
    print(f"Features removed: {set(features_before) - set(X.columns)}")

    # Quantile transformer
    transformer = QuantileTransformer()
    X = transformer.fit_transform(X, y)

    # Standardize the features
    scaler = RobustScaler()
    X = scaler.fit_transform(X, y)
    return X, y


def get_train_test_tensors():
    X, y = prepare_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.long)
    y_test = torch.tensor(y_test.values, dtype=torch.long)
    return X_train, X_test, y_train, y_test


def plot_confussion_matrix(predictions, labels, save_path=None):
    cf_matrix = confusion_matrix(labels.numpy(), predictions.numpy())
    plt.figure(figsize=(10, 10))
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        title = save_path.split("/")[-1].split(".")[0].replace("_", " ")
        print(title)
        plt.title(title)
        plt.savefig(save_path, dpi=300)
    plt.show()

def plot_loss(train_losses, save_path=None):
    plt.plot(train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    if save_path:
        title = save_path.split("/")[-1].split(".")[0].replace("_", " ")
        print(title)
        plt.title(title)
        plt.savefig(save_path, dpi=300)
    plt.show()