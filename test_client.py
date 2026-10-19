import requests

url = 'http://localhost:8000/prediction_batch'

data = {"features": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]}

response = requests.post(url, json=data)

if response.status_code == 200:
    print(response.json())
else:
    print(f"Server Error ({response.status_code}):")
    print(response.text)  # Prints the full error or HTML returned by Flask