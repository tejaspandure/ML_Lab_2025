
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

INPUT_PATH = "bank_cleaned.csv"
OUTPUT_PATH = "bank_cleaned.csv"


HEADERS = ["age","job","material","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]

def read_data(path):
    return pd.read_csv("bank_cleaned.csv")

def get_headers(dataset):
    return dataset.columns.values

def add_headers(dataset, headers):
    dataset.columns = headers
    return dataset

def encode_categorical_features(dataset, categorical_columns):
    encoder = LabelEncoder()
    for iCnt in categorical_columns:
        dataset[iCnt] = encoder.fit_transform(dataset[iCnt])
    return dataset

def data_file_to_csv():
    dataset = read_data(INPUT_PATH)
    dataset = add_headers(dataset, HEADERS)
    dataset.to_csv(OUTPUT_PATH, index=False)
    print("File saved...!")

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    train_x,test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header], train_size=train_percentage)
    return train_x, test_x, train_y, test_y

def handel_missing_values(dataset, missing_values_header, missing_label):
    return dataset[dataset[missing_values_header] != missing_label]

def random_forest_classifier(features, target):
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def dataset_statistics(datset):
    print(datset.describe())
    return dataset_statistics

def main():
    dataset = pd.read_csv(OUTPUT_PATH)
    dataset_statistics(dataset)
    dataset = handel_missing_values(dataset, HEADERS[6], '?')
    
    categorical_columns = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome", "y"]
    dataset = encode_categorical_features(dataset, categorical_columns)
    
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, HEADERS[1:-1], HEADERS[-1])
    
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)
    
    trained_model = random_forest_classifier(train_x, train_y)
    print("Trained model :: ", trained_model)
    predictions = trained_model.predict(test_x)
    
    print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
    print("Test Accuracy :: ", accuracy_score(test_y, predictions))
    print("Confusion matrix", confusion_matrix(test_y, predictions))

if __name__ == "__main__":
    main()

