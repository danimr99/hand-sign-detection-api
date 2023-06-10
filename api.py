from flask import Flask, request
from flask_cors import CORS
from hands_detection import HandsDetection
import json


# Define constants
HANDS_DETECTION_MODEL_PATH = './hand-sign-models/cat/model.pickle'
HANDS_DETECTION_CLASSES_PATH = './hand-sign-models/cat/labels.json'

# Create a Flask app
app = Flask(__name__)

# Enable CORS
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

@app.route('/api/detect', methods=['POST'])
def predict_hand_sign():
    # Check if the request contains the Content-Type header
    if not 'Content-Type' in request.headers:
        error_msg = {'error': 'Content-Type must be application/json'}
        return json.dumps(error_msg), 400

    # Get the Content-Type header
    content_type = request.headers['Content-Type']

    # Check if the Content-Type header is application/json
    if content_type == 'application/json':
        # Check if the request contains JSON data
        if request.is_json:
            try:
                # Get hands landmarks from the request body
                hands = request.get_json()

                # Check if hands_landmarks is a list
                if not isinstance(hands, list):
                    error_msg = {
                        'message': 'Request body must be a list of Mediapipe Hands'}
                    return json.dumps(error_msg), 400

                # Check if data contains between 1 and 2 hands
                if len(hands) == 0 or len(hands) > 2:
                    error_msg = {
                        'message': 'You must provide 1 or 2 Mediapipe Hands'}
                    return json.dumps(error_msg), 400

            except:
                error_msg = {'error': 'Invalid JSON data'}
                return json.dumps(error_msg), 400

            # Create a HandsDetection object
            hands_detection = HandsDetection(
                HANDS_DETECTION_MODEL_PATH, HANDS_DETECTION_CLASSES_PATH)

            # Process hands landmarks
            hands_landmarks = hands_detection.process_landmarks(hands)

            # Predict hand sign
            return hands_detection.predict(hands_landmarks)
        else:
            error_msg = {'error': 'Request must contain JSON data'}
            return json.dumps(error_msg), 400
    else:
        error_msg = {'error': 'Content-Type must be application/json'}
        return json.dumps(error_msg), 400


if __name__ == '__main__':
    app.run(port=8000, debug=True)
