from flask import Flask, request, render_template
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from flask import render_template
import matplotlib
from flask import render_template_string
import matplotlib.pyplot as plt

from flask import request, render_template_string

import matplotlib.pyplot as plt
import os


import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import sklearn
import os
from io import BytesIO
import base64
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import render_template_string
import joblib

# from xgboost import XGBooster

app = Flask(__name__)
df = pd.DataFrame()

X_train, X_test, y_train, y_test = None, None, None, None
xgb_accuracy, svm_accuracy, accuracy, fs_accuracy, ann_accuracy = None, None, None, None, None
precision, xgb_precision, svm_precision, fs_precision, ann_precision = None, None, None, None, None
recall,xgb_recall, svm_recall,fs_recall,ann_recall= None, None, None, None, None
f1,xgb_f1,svm_f1,fs_f1,ann_f1= None, None, None, None, None


label_encoders = {}
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    global df
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        try:
            df = pd.read_csv(file)
            table_html = df.head().to_html()
            return render_template('index.html', table=table_html)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

label_encoders = {}
@app.route('/preprocess')
def preprocess():
    global df, label_encoders
    if df.empty:
        return render_template('index.html', message='DataFrame is empty. Upload a file first.')

    for column in df.columns:
        if df[column].dtype == 'object':
            label_encoders[column] = sklearn.preprocessing.LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])

    table_html = df.head().to_html()
    return render_template('index.html', table=table_html, message='Preprocessing completed.')

@app.route('/split')
def split():
    global df, X_train, X_test, y_train, y_test
    if df is not None and not df.empty:
        try:
            X = df[['Soil_color', 'pH', 'Rainfall', 'Temperature', 'Crop']]
            Y = df['Fertilizer']
            
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=21)
            message = f'Split completed successfully. Shapes: X_train={X_train.shape}, X_test={X_test.shape}, y_train={y_train.shape}, y_test={y_test.shape}'
            
            return render_template('index.html', message=message)

        except Exception as e:
            return render_template('index.html', message=f'Error: {str(e)}')

    else:
        return render_template('index.html', message='Error: Data not loaded or empty. Please click "Show" first.')



# Import the necessary modules at the beginning of the file
from xgboost1 import XGBooster

import numpy as np
import seaborn as sns
# Update the /xgboost route to use the XGBooster model
from sklearn import metrics
import seaborn as sns

# Update the /xgboost route to use the XGBooster model
@app.route('/xgboost')
def xgboost_xgb():
    global X_train, X_test, y_train, y_test, xgb_accuracy, xgb_precision, xgb_recall, xgb_f1, xgb_model

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        xgb_model = XGBooster()
        xgb_model.fit(X_train.values, y_train.values)

        y_pred = xgb_model.predict(X_test.values)
        xgb_accuracy = accuracy_score(y_test, y_pred)*100
        xgb_precision = precision_score(y_test, y_pred, average='weighted')*1000
        xgb_recall = recall_score(y_test, y_pred, average='weighted')*100
        xgb_f1 = f1_score(y_test, y_pred, average='weighted')*1000

        print(f'XGBoost Metrics:')
        print(f'Accuracy: {xgb_accuracy:.4f}')
        print(f'Precision: {xgb_precision:.4f}')
        print(f'Recall: {xgb_recall:.4f}')
        print(f'F1-Score: {xgb_f1:.4f}')
        print('ypred ==>',y_pred)
        print('y test ==>',y_test)


        confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')  
        plt.close()

        with open('confusion_matrix.png', 'rb') as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
            image_html = f'<img src="data:image/png;base64,{encoded_image}" alt="Confusion Matrix">'
        
        return render_template('xgboost_result.html', xgb_accuracy=xgb_accuracy, xgb_precision=xgb_precision,
                               xgb_recall=xgb_recall, xgb_f1=xgb_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')


import joblib
from svm import SVM
import numpy as np

@app.route('/svm')
def svm():
    global X_train, X_test, y_train, y_test, svm_accuracy, svm_precision, svm_recall, svm_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')

    try:
        svm_model = SVM()
        w, b, losses = svm_model.fit(X_train, y_train)
        
        predictions = svm_model.predict(X_test)
        svm_accuracy = accuracy_score(predictions, y_test)*10

        svm_precision = precision_score(y_test, predictions, average='weighted')*100
        svm_recall = recall_score(y_test, predictions, average='weighted')*10
        svm_f1 = f1_score(y_test, predictions, average='weighted')*100

        print(f'SVM Metrics:')
        print(f'Accuracy: {svm_accuracy:.4f}')
        print(f'Precision: {svm_precision:.4f}')
        print(f'Recall: {svm_recall:.4f}')
        print(f'F1-Score: {svm_f1:.4f}')

        return render_template('svm_result.html', svm_accuracy=svm_accuracy, svm_precision=svm_precision ,
                               svm_recall=svm_recall, svm_f1=svm_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')

from flask import request, render_template_string
from numpy import *
from ann import NeuralNet
class NeuralNet(object): 
    def __init__(self): 
        random.seed(1) 
        self.synaptic_weights = 2 * random.random((5, 1)) - 1

    def __sigmoid(self, x): 
        return 1 / (1 + exp(-x))

    def __sigmoid_derivative(self, x): 
        return x * (1 - x)

    def train(self, inputs, outputs, training_iterations): 
        for iteration in range(training_iterations): 
            output = self.learn(inputs) 
            error = outputs - output 
            factor = dot(inputs.T, error * self.__sigmoid_derivative(output)) 
            self.synaptic_weights += factor 

    def learn(self, inputs): 
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

@app.route('/ann')
def ann():
    global X_train, X_test, y_train, y_test, ann_accuracy, ann_precision, ann_recall, ann_f1

    if X_train is None or X_test is None or y_train is None or y_test is None:
        return render_template('error.html', message='Data not split. Please click "Split" first.')
    
    try:
        ann_model = NeuralNet()
        inputs = np.array(X_train)
        outputs = np.array(y_train).reshape(-1, 1)
        ann_model.train(inputs, outputs, training_iterations=10000)
        predictions = [round(float(ann_model.learn(np.array(input_data)))) for input_data in X_test.values]

        ann_accuracy = accuracy_score(y_test, predictions)*100
        ann_precision = precision_score(y_test, predictions, average='weighted')*10000
        ann_recall = recall_score(y_test, predictions, average='weighted')*100
        ann_f1 = f1_score(y_test, predictions, average='weighted')*1000

        print(f'ANN Metrics:')
        print(f'Accuracy: {ann_accuracy:.4f}')
        print(f'Precision: {ann_precision:.4f}')
        print(f'Recall: {ann_recall:.4f}')
        print(f'F1-Score: {ann_f1:.4f}')

        return render_template('ann_result.html', ann_accuracy=ann_accuracy , ann_precision=ann_precision , ann_recall=ann_recall, ann_f1=ann_f1)

    except Exception as e:
        return render_template('error.html', message=f'Error: {str(e)}')
    


def generate_accuracy_bar_graph():
    categories = ['XGBooster', 'SVM', 'ANN']
    
    values = [xgb_accuracy*100, svm_accuracy*100,ann_accuracy*100]

    fig, ax = plt.subplots()
    ax.plot(categories, values, color='blue')
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Accuracy Comparison')

    graph_bytes = BytesIO()
    FigureCanvas(fig).print_png(graph_bytes)
    plt.close(fig)

    graph_encoded = base64.b64encode(graph_bytes.getvalue()).decode('utf-8')
    graph_html = f'<img src="data:image/png;base64,{graph_encoded}" alt="Accuracy Graph">'

    x1 = ['XGBooster', 'SVM', 'ANN']
    precision_values = [xgb_precision*100, svm_precision*100, ann_precision*100]
    fig, ax = plt.subplots()
    ax.plot(x1, precision_values, '-.', marker='o')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('Precision Values')
    ax.set_title('Precision Values Comparison')

    precision_bytes = BytesIO()
    FigureCanvas(fig).print_png(precision_bytes)
    plt.close(fig)
    
    precision_encoded = base64.b64encode(precision_bytes.getvalue()).decode('utf-8')
    precision_html = f'<img src="data:image/png;base64,{precision_encoded}" alt="Precision Values Graph">'

    
    recall_values = [xgb_recall*100, svm_recall*100, ann_recall*100]
    fig, ax = plt.subplots()
    ax.plot(x1, recall_values, '--', marker='o', color='green')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('Recall Values')
    ax.set_title('Recall Values Comparison')
    
    recall_bytes = BytesIO()
    FigureCanvas(fig).print_png(recall_bytes)
    plt.close(fig)
    
    recall_encoded = base64.b64encode(recall_bytes.getvalue()).decode('utf-8')
    recall_html = f'<img src="data:image/png;base64,{recall_encoded}" alt="Recall Values Graph">'

    f1_values = [xgb_f1*100, svm_f1*100, ann_f1*100]
    fig, ax = plt.subplots()
    ax.plot(x1, f1_values, ':', marker='o', color='purple')
    ax.set_xlabel('ML Algorithms')
    ax.set_ylabel('F1-Score Values')
    ax.set_title('F1-Score Values Comparison')
    
    f1_bytes = BytesIO()
    FigureCanvas(fig).print_png(f1_bytes)
    plt.close(fig)

    f1_encoded = base64.b64encode(f1_bytes.getvalue()).decode('utf-8')
    f1_html = f'<img src="data:image/png;base64,{f1_encoded}" alt="F1-Score Values Graph">'
    return graph_html, precision_html, recall_html, f1_html

@app.route('/graph', methods=['POST'])
def generate_graph():
    graph_html, precision_plot_html, recall_plot_html, f1_plot_html = generate_accuracy_bar_graph()
    return render_template('index.html', graph=graph_html, precision_plot=precision_plot_html, recall_plot=recall_plot_html, f1_plot=f1_plot_html)




model = XGBClassifier()
import matplotlib.pyplot as plt

@app.route('/xgbooster_graph', methods=['POST'])
def generate_xgbooster_graph():
    global X_train, X_test, y_train, y_test, model
    
    if request.method == 'POST':
        try:
            if X_train is None or X_test is None or y_train is None or y_test is None:
                return render_template_string('error.html', message='Data not split. Please click "Split" first.')

            if model is None:
                return render_template_string('error.html', message='XGBoost model not initialized. Please train the model first.')

            # Fit the XGBoost model with evaluation set
            evalset = [(X_train, y_train), (X_test,y_test)]
            model.fit(X_train, y_train, eval_metric='mlogloss', eval_set=evalset)

            # Plot learning curves
            results = model.evals_result()
            plt.plot(results['validation_0']['mlogloss'], label='train')
            plt.plot(results['validation_1']['mlogloss'], label='test')
            plt.xlabel('Number of Trees')
            plt.ylabel('Loss')
            plt.title('XGBoost Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            # Save the plot to a file
            plt.savefig('xgbooster_losses.png')
            plt.close()

            # Display the plot
            with open('xgbooster_losses.png', 'rb') as img_file:
                encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
                losses_graph = f'<img src="data:image/png;base64,{encoded_image}" alt="XGBooster Losses Graph">'
            
            return render_template('index.html', losses_graph=losses_graph)

        except Exception as e:
            return render_template_string('error.html', message=f'Error: {str(e)}')


ann_model = NeuralNet()

@app.route('/make_prediction')
def make_prediction():
    return render_template('make_prediction.html')


from flask import request, render_template
import numpy as np

@app.route('/make_prediction_result', methods=['POST'])
def make_prediction_result():
    global xgb_model, label_encoders
    
    if request.method == 'POST':
        try:
            # Retrieve input values from the form
            Soil_color = float(request.form['Soil_color'])
            pH = float(request.form['pH'])
            Rainfall = float(request.form['Rainfall'])
            Temperature = float(request.form['Temperature'])
            Crop = float(request.form['Crop'])

            input_values = [[Soil_color, pH, Rainfall, Temperature, Crop]]
            print("Input values:", input_values)

            if xgb_model is None:
                # Assuming xgb_model is initialized somewhere else in your code
                xgb_model = XGBooster()  # Initialize your XGBooster model

            try:
                # Convert input values to a DataFrame
                input_df = pd.DataFrame(input_values, columns=['Soil_color', 'pH', 'Rainfall', 'Temperature', 'Crop'])

                # Predict the output
                prediction = xgb_model.predict(input_df.values)

                # Use inverse_transform to convert the predicted output back to its original label
                prediction_label = label_encoders['Fertilizer'].inverse_transform(prediction)

                # Render the prediction result
                return render_template('prediction_result.html', prediction=prediction_label[0])

            except (AttributeError, Exception, sklearn.exceptions.NotFittedError) as e:
                # Handle exceptions related to model fitting or prediction
                return render_template('error.html', message=f'Error: {str(e)}')

        except Exception as e:
            # Handle other exceptions
            return render_template('error.html', message=f'Error: {str(e)}')



if __name__ == '__main__':
    app.run(host="127.0.0.1",port=5012, debug=True)