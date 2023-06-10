import pickle
import json
import numpy as np


class HandsDetection:
    NUM_LANDMARKS = 21
    NUM_COORDINATES = 2

    def __init__(self, model_file_path, classes_file_path):
        # Load model
        self.model = pickle.load(open(model_file_path, 'rb'))['model']

        # Read json file with classes
        with open(classes_file_path, 'r') as f:
            self.classes = json.load(f)

    def process_landmarks(self, hands):
        aux = []
        x_ = []
        y_ = []

        # Normalize landmarks
        for hand in hands:
            for i in range(len(hand['keypoints'])):
                x = hand['keypoints'][i]['x']
                y = hand['keypoints'][i]['y']

                x_.append(x)
                y_.append(y)

            for i in range(len(hand['keypoints'])):
                x = hand['keypoints'][i]['x']
                y = hand['keypoints'][i]['y']

                # Normalize coordinates
                normalized_x = (x - min(x_)) / (max(x_) - min(x_))
                normalized_y = (y - min(y_)) / (max(y_) - min(y_))

                aux.append(normalized_x)
                aux.append(normalized_y)

            # Append zeros if sign uses one hand only
            if len(hands) == 1:
                for i in range((HandsDetection.NUM_LANDMARKS * HandsDetection.NUM_COORDINATES)):
                    aux.append(0)

        return aux

    def predict(self, vectorized_landmarks):
        predictions = self.model.predict_proba(
            [np.asarray(vectorized_landmarks)])

        # Create a dictionary to store the class probabilities
        class_probabilities = []

        # Iterate over the classes and their respective identifiers
        for class_name, class_id in self.classes.items():
            # Get the probability prediction for the current class
            class_probability = predictions[:, class_id]

            # Check if the class probability is not zero
            if class_probability[0] > 0:
                # Add the class and its probability to the dictionary as a new JSON object
                class_probabilities.append({
                    'label': class_name,
                    'confidence': class_probability[0]
                })

        # Create the final JSON object
        return json.dumps(class_probabilities)
