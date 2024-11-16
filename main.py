import asyncio
import websockets
import json
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Load a sample dataset (Iris dataset for this example)
def load_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Train a simple model (RandomForest)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and scaler
    joblib.dump(model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, scaler

# Predict function to use the model and scaler
def predict(model, scaler, data):
    data = np.array(data).reshape(1, -1)  # Ensure it's in the correct shape
    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    return prediction[0]

# WebSocket server to interact with the model
async def handler(websocket, path):
    print("Client connected")
    
    # Load the model and scaler (one-time load)
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')

    try:
        while True:
            # Receive message from client
            message = await websocket.recv()
            print(f"Received message: {message}")

            # Parse the incoming message (assuming it's a JSON string)
            try:
                data = json.loads(message)
                if "features" in data:
                    # Make prediction
                    features = data["features"]
                    prediction = predict(model, scaler, features)
                    
                    # Send back the prediction result
                    response = {"prediction": int(prediction)}
                    await websocket.send(json.dumps(response))
                else:
                    await websocket.send(json.dumps({"error": "Invalid input format."}))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"error": "Invalid JSON format."}))
    
    except websockets.ConnectionClosed:
        print("Client disconnected")

# Start WebSocket server
async def main():
    # Create WebSocket server, listening on port 8765
    server = await websockets.serve(handler, "localhost", 8765)
    print("WebSocket server started on ws://localhost:8765")
    
    # Keep the server running
    await server.wait_closed()

if __name__ == "__main__":
    # Train model once before starting the server
    load_model()
    
    # Run the WebSocket server
    asyncio.run(main())
