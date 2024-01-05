from flask import Flask, request, jsonify
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

app = Flask(__name__)


def load_data(file_path):
    return pd.read_csv(file_path)


def filter_data(df, filters):
    filtered_data = df.copy()
    for column, value in filters.items():
        filtered_data = filtered_data[filtered_data[column] == value]
    return filtered_data


def convert_to_json(df):
    return df.to_json(orient='records', lines=True)


def load_test_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def generate_classification_report(y_test, y_pred):
    class_report = classification_report(y_test, y_pred, zero_division=1)
    print("Classification Report:")
    print(class_report)


def predict_model(table):
    target_column = 'series_id'
    df = table.copy()
    df.replace('?', pd.NA, inplace=True)

    categorical_columns = ['painkillers', 'gender', 'age', 'time_from_injury', 'know_nlp', 'faith_nlp',
                           'series_time_1', 'movie_time_1']

    for column in categorical_columns:
        df[column] = df[column].astype('category')

    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy}")
    generate_classification_report(y_test, y_pred)


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Retrieve all filters from URL parameters
        filters = {}
        for key, value in request.args.items():
            filters[key] = value

        df = load_data('mindmover-trainingset-4.csv')
        filtered_data = filter_data(df, filters)
        predict_model(df)

        if filtered_data.empty:
            return jsonify({"result": "No matching records found"})
        else:
            result = convert_to_json(filtered_data)
            return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)})



if __name__ == '__main__':
    app.run(debug=True)
