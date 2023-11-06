# Project Decription
## Problem Statement
The primary problem to address is predicting machine failures in the milling process. The challenge is to build a machine learning model that can accurately forecast when a machine will fail based on the given features. This is critical for maintenance scheduling, reducing downtime, improving productivity, and ensuring the quality of the products.

The complexity of the problem is compounded by the fact that the 'machine failure' label does not indicate which of the five independent failure modes caused the failure. The machine learning model must discern patterns in the data that lead to failures without explicit failure mode labels.
## Proposed Solution
The proposed solution involves developing a predictive maintenance machine learning model, specifically a binary classification model, to forecast machine failures based on historical data from a milling machine. Using features such as air and process temperatures, rotational speed, torque, tool wear, and product quality indicators, the model will predict the binary outcome of machine failure. This classification task falls under supervised learning, as it will use labeled training data (where the outcomes are known) to learn the patterns associated with failures. To determine the most influential features, the model's feature importances will be analyzed post-training, which will reveal which variables have the most significant impact on predicting failure. Techniques such as Random Forest or Gradient Boosting, known for providing insights on feature importance, will be used. These insights can then inform maintenance decisions, leading to more effective failure prevention strategies.

# Download Dataset
- The [dataset](https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020) can be downloaded using command below
```
wget -O data/raw_data.csv https://raw.githubusercontent.com/Edoar-do/AI4I-Predictive-Maintenance/main/ai4i2020.csv
```
# Depedency & Environment Management
### Activate Virtual Environment & Install Dependencies
1. Activate the virtual environment using the appropriate command for your operating system:
 - **Windows**
```
.venv\Scripts\activate
```
   - **MacOS and Linux**
```
source .venv/bin/activate
```
2. Upgrade PIP first using this command
```
python -m pip install --upgrade pip
``` 
3. Install packages using requirements.txt
```
pip install -r requirements.txt
```
4. Deactivate the virtual environment (optional)
```
deactivate
```
### Python Version Using in This Project
Python 3.10.11

# Containerization
Here are the steps and commands to create a Docker image and run a Docker container:
1. Open Terminal: First, you need to open a terminal on your machine.
2. Navigate to Your Project Directory: Use the cd command to navigate to the directory containing your project files.
```bash
cd .\source\
```
3. Build the Docker Image: You will use the docker build command to create an image. Replace your_image_name with the name you want to give your Docker image.
```bash
docker build -t your_image_name .
```
4. Check the Image: After building the image, you can see it listed by running:
```bash
docker images
```
5. Run the Docker Container: Now you can run a container from the image you just built. Replace your_container_name with the name you want to give your Docker container. Also, if your application needs specific ports to be open or environment variables, make sure to specify those with the -p and -e flags respectively.
```bash
docker run --name your_container_name -p 8080:8080 -d your_image_name
```

# Project Structure
```
ML ZoomCamp Midterm Project
│
├───data
│       raw_data.csv
│
├───model
│       LogisticRegression_model.joblib
│       logistic_regression_model.joblib
│
├───notebook
│       EDA.ipynb
│       ML_model.ipynb
│
└───source
    │   app.py
    │   Dockerfile
    │   logistic_regression_model.joblib
    │   ml_pipeline.py
    │   predict.py
    │
    └───modules
         viz.py
```

# Project Evaluation Criteria
- [x] Problem Description
- [x] EDA
- [x] Model Training
- [x] Exporting to Script
- [x] Reproducibility
- [x] Model Deployment
- [x] Dependency and environment management
- [x] Containerization
- [ ] Cloud Deployment