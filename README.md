# Hand Sign Detection API

This repository contains the implementation of a Flask REST API that uses a SciKit RandomForestClassifier pretrained for hand sign detection using Mediapipe Hands.

## USAGE

1. Install dependencies

	**NOTE**: You can clone the environment from [environment.yml](https://github.com/danimr99/hand-sign-detection-api/blob/main/environment.yml) using Conda.

  	```console
		$ conda env create -f environment.yml
	```

  	It will create a Conda environment with the name specified in the environment.yml file.
	
2. On the root directory of this project, create a folder to add your machine learning models to use.
  You must add:
    - labels.json
    - model.pickle

3. Run the server using the following command:

  	```console
		$ python3 api.py
	```
