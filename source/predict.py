import requests

# Define the URL where the Flask app is hosted, adjust the port if necessary
url = "http://localhost:5000/predict"

# Define the dummy data to be sent for prediction
# The keys should match the column names expected by your model
client = {
    "UDI": 90,
    "Product ID": "M14949",
    "Type": "M",
    "Air temperature [K]": 298.9,
    "Process temperature [K]": 308.9,
    "Rotational speed [rpm]": 1487,
    "Torque [Nm]": 39.5,
    "Tool wear [min]": 30,
    "Machine failure": 0,
    "TWF": 0,
    "HDF": 0,
    "PWF": 0,
    "OSF": 0,
    "RNF": 0
}

# Send the POST request to the Flask app with the dummy data as JSON
response = requests.post(url, json=client)

# Parse the JSON response
result = response.json()  # Assuming the result is a JSON object containing the prediction

# Print the result
print("The prediction for the provided data is:", result)
