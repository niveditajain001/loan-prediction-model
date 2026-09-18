# Bank Loan Prediction AI

## Overview
This project is an end-to-end Machine Learning pipeline designed to predict bank loan approvals. By analyzing historical applicant data (such as income, education, and credit history), the system identifies mathematical patterns that correlate with loan repayment and defaults. The project features a Random Forest Classifier trained on cleaned data and a Streamlit web application that provides real-time, data-driven approval decisions for new applicants.

## Features
* **Modular Architecture:** Clean separation between the offline training environment (`train.py`), data preprocessing (`preprocess.py`), and the live production web application (`app.py`).
* **Real-Time Inference:** The web UI instantly maps categorical user inputs (e.g., "Male", "Urban") to numerical arrays and passes them to the serialized machine learning model for immediate prediction.
* **Automated Evaluation Metrics:** The training script automatically generates and prints the accuracy score, confusion matrix, and classification report to the terminal to validate model performance on unseen data.

## Technologies & Tools Used
* **Python 3:** Core programming language.
* **Pandas & NumPy:** Data cleaning, manipulation, and array structuring.
* **Scikit-Learn:** Machine learning model training (Random Forest) and metric evaluation.
* **Streamlit:** Web application framework for the interactive front-end UI.
* **Joblib:** Object serialization for exporting and loading the trained model.
* **Git/GitHub:** Version control and repository hosting.

## Steps to Install & Run the Project
This project is fully executable via the command line. Follow these exact steps to set up the environment and run the pipeline.

**1. Clone the Repository**
Open your terminal and clone the project to your local machine:
```bash
 git clone https://github.com/niveditajain001/loan-prediction-model.git
cd loan-prediction-model
```

**2. Set Up a Virtual Environment**
It is highly recommended to run this project inside an isolated virtual environment to prevent dependency conflicts:
```bash
python -m venv venv
```

**3. Activate the Virtual Environment**
* On **Windows**:
  ```bash
  venv\Scripts\activate
  ```
* On **macOS/Linux**:
  ```bash
  source venv/bin/activate
  ```

**4. Install Dependencies**
With the virtual environment active, install the required packages:
```bash
pip install pandas numpy scikit-learn streamlit joblib seaborn matplotlib
```

**5. Execute the Training Pipeline**
Run the training script to clean the data, train the Random Forest model, view the evaluation metrics, and export the `loan_model.pkl` file:
```bash
python train.py
```

**6. Launch the Web Application**
Once the model is successfully saved, start the Streamlit server to interact with the UI:
```bash
streamlit run app.py
```

## Instructions for Testing
1. After running the `streamlit run app.py` command, a local URL (usually `http://localhost:8501`) will appear in your terminal. Open this link in your web browser.
2. The UI will display a series of dropdowns and number inputs. Enter a sample applicant profile with strong financial metrics (e.g., Graduate, Good Credit History, Applicant Income: 8000, Loan Amount: 100).
3. Click the **"Predict Loan Status"** button. Verify that the system processes the inputs and displays "🎉 LOAN APPROVED!".
4. Change the parameters to reflect poor financial health (e.g., Bad Credit History, High Loan Amount, Low Income). Click the button again to verify the system accurately switches its prediction to "❌ LOAN REJECTED."

## Screenshots
*(Add your screenshots to the repository folder and update these image paths)*
* **Terminal Execution & Model Metrics:** `./imges/terminal_output.png`
* **Streamlit Web Application UI:** `./images/web_app.png`