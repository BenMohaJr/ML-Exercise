import requests

# Define the URL with the appropriate filters as parameters
url = 'http://127.0.0.1:5000/predict'
params = {
    "painkillers": "No",
    "gender": "M",
    "age": "25-50",
    "time_from_injury": "1-3",
    "know_nlp": "Y",
    "faith_nlp": "Y"
}

# Send the GET request
response = requests.get(url, params=params)

# Check the response status and print the result
if response.status_code == 200:
    result = response.json().get('result')
    print(f"Result: {result}")
else:
    print(f"Error: {response.status_code}, {response.text}")
