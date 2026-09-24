from flask import Flask, render_template, request, send_file
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import seaborn as sns
import io
import base64
import os

# Paths are resolved relative to this file so the app can be started from anywhere.
HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, 'models')
os.chdir(HERE)

df1 = pd.read_csv('HQprecipitation Data Test.csv')
df2 = pd.read_csv('precipitationCal Data Test.csv')
nowImergL = datetime.now() - timedelta(days=1, hours=22)

if datetime.date(pd.to_datetime(df1['time'][0])) == datetime.date(nowImergL) or datetime.date(pd.to_datetime(df2['time'][0])) == datetime.date(nowImergL) :
    pass
else:
    import imerg_scraper  # V07 fetcher, see scripts/imerg_client.py


app = Flask(__name__)

# importing machine learning model and data
def load_model(name):
    # The 2023 pickles were saved with scikit-learn 1.2. The Decision Trees do not
    # load under newer versions; the app keeps running and reports it instead.
    try:
        return pickle.load(open(os.path.join(MODELS, name), 'rb'))
    except Exception as exc:
        print(f'[warn] could not load {name}: {type(exc).__name__}. Pin scikit-learn==1.2.2 or retrain with scripts/train_v2.py')
        return None

HQBayesModel = load_model('HQBayesModel.pkl')
HQLogRegModel = load_model('HQLogRegModel.pkl')
HQDtModel = load_model('HQDtModel.pkl')
HQSvmachineModel = load_model('HQSvmachineModel.pkl')

PrecipCalBayesModel = load_model('PrecipCalBayesModel.pkl')
PrecipCalDtModel = load_model('PrecipCalDtModel.pkl')
PrecipCalLogRegModel = load_model('PrecipCalLogRegModel.pkl')
PrecipCalSvmachineModel = load_model('PrecipCalSvmachineModel.pkl')

FreshDataHQ = pd.read_csv('HQprecipitation Data Test.csv')
FreshDataCal = pd.read_csv('precipitationCal Data Test.csv')

# cleaning data for high quality precipitation data
FreshDataHQ = FreshDataHQ.drop(columns=['time'])
column_list = []
for i in range(225):
    colName = 'HqPrecip_'+str(i)
    column_list.append(colName)
column = []
for i in range(1):
    values = (FreshDataHQ['HqPrecips'][i])[1:-1].split(', ')
    for j in range(225):
        values[j] = float(values[j])
    column.append(values)

FreshDataVectorHQ = pd.DataFrame(np.column_stack(list(zip(*column))), columns=column_list)
arrHQ = FreshDataVectorHQ.iloc[0].to_numpy().reshape(15,15)

# cleaning data for precipitation calibrated data
FreshDataCal = FreshDataCal.drop(columns=['time'])
column_list = []
for i in range(225):
    colName = 'precipitationCal_'+str(i)
    column_list.append(colName)
column = []
for i in range(1):
    values = (FreshDataCal['PrecipCals'][i])[1:-1].split(', ')
    for j in range(225):
        values[j] = float(values[j])
    column.append(values)

FreshDataVectorCal = pd.DataFrame(np.column_stack(list(zip(*column))), columns=column_list)
arrCal = FreshDataVectorCal.iloc[0].to_numpy().reshape(15,15)

# application functions
@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    _chosen = {
        'Naive Bayes Model (HQ)': HQBayesModel, 'Logistic Regression Model (HQ)': HQLogRegModel,
        'Decision Tree Model (HQ)': HQDtModel, 'Support Vector Machine Model (HQ)': HQSvmachineModel,
        'Naive Bayes Model (PrecipCal)': PrecipCalBayesModel, 'Logistic Regression Model (PrecipCal)': PrecipCalLogRegModel,
        'Decision Tree Model (PrecipCal)': PrecipCalDtModel, 'Support Vector Machine Model (PrecipCal)': PrecipCalSvmachineModel,
    }.get(request.form['models'], 'unknown')
    if _chosen is None:
        result = request.form['models'] + ' could not be loaded in this environment (see console). Pin scikit-learn==1.2.2 or retrain with scripts/train_v2.py.'
        return render_template('index.html', **locals())
# CODE FOR HIGH QUALITY PRECIPITATION MODELS
    # Code for handling the naive bayes model with high quality precipitation data
    if request.form['models'] == 'Naive Bayes Model (HQ)':
        result = HQBayesModel.predict(FreshDataVectorHQ)
        if result == 0:
            result = 'Naive Bayes Model (HQ) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Naive Bayes Model (HQ) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())

    # Code for handling the logistic regression model with high quality precipitation data
    if request.form['models'] == 'Logistic Regression Model (HQ)':
        result = HQLogRegModel.predict(FreshDataVectorHQ)
        if result == 0:
            result = 'Logistic Regression Model (HQ) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Logistic Regression Model (HQ) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())

    # Code for handling the decision tree model with high quality precipitation data
    if request.form['models'] == 'Decision Tree Model (HQ)':
        result = HQDtModel.predict(FreshDataVectorHQ)
        if result == 0:
            result = 'Decision Tree Model (HQ) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Decision Tree Model (HQ) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        
    # Code for handling the support vector machine model with high quality precipitation data
    if request.form['models'] == 'Support Vector Machine Model (HQ)':
        result = HQSvmachineModel.predict(FreshDataVectorHQ)
        if result == 0:
            result = 'Support Vector Machine Model (HQ) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Support Vector Machine Model (HQ) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        
# CODE FOR PRECIPITATION CALIBRATED MODELS
    # Code for handling the naive bayes model with precipitation calibrated data
    if request.form['models'] == 'Naive Bayes Model (PrecipCal)':
        result = PrecipCalBayesModel.predict(FreshDataVectorCal)
        if result == 0:
            result = 'Naive Bayes Model (PrecipCal) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Naive Bayes Model (PrecipCal) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())

    # Code for handling the logistic regression model with precipitation calibrated data
    if request.form['models'] == 'Logistic Regression Model (PrecipCal)':
        result = PrecipCalLogRegModel.predict(FreshDataVectorCal)
        if result == 0:
            result = 'Logistic Regression Model (PrecipCal) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Logistic Regression Model (PrecipCal) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())

    # Code for handling the decision tree model with precipitation calibrated data
    if request.form['models'] == 'Decision Tree Model (PrecipCal)':
        result = PrecipCalDtModel.predict(FreshDataVectorCal)
        if result == 0:
            result = 'Decision Tree Model (PrecipCal) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Decision Tree Model (PrecipCal) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        
    # Code for handling the support vector machine model with precipitation calibrated data
    if request.form['models'] == 'Support Vector Machine Model (PrecipCal)':
        result = PrecipCalSvmachineModel.predict(FreshDataVectorCal)
        if result == 0:
            result = 'Support Vector Machine Model (PrecipCal) predicts no flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())
        else:
            result = 'Support Vector Machine Model (PrecipCal) predicts flooding on ' + str(datetime.date(nowImergL))
            return render_template('index.html', **locals())

@app.route('/visualize', methods=['POST', 'GET'])
def visualize():
    # Defining plot area and reading in pre-cropped image of selangor
    fig, ax = plt.subplots(figsize=(6,6))
    map_img = plt.imread('SelangorMap.png') 

    # Changing the data in the generated map based on model chosen by user
    if 'High Quality Precipitation' in request.form:
        data = 'High Quality Precipitation'
        hmax = sns.heatmap(arrHQ,
                cmap = 'Blues',
                alpha = 0.4,
                annot = False,
                zorder = 2,
                )
        # Displaying the generated map
        hmax.imshow(map_img,
            aspect = hmax.get_aspect(),
            extent = hmax.get_xlim() + hmax.get_ylim(),
            zorder = 1)
        
        # Grabbing generated map from IO buffer and rendering it on html site
        canvas = FigureCanvas(fig)
        img = io.BytesIO()
        plt.savefig(img)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        source = 'data:image/png;base64,{}'.format(plot_url)
        return render_template('index.html', **locals())
    
    else:
        data = 'Precipitation Calibrated'
        hmax = sns.heatmap(arrCal,
                cmap = 'Blues',
                alpha = 0.4,
                annot = False,
                zorder = 2,
                )        
        # Displaying the generated map
        hmax.imshow(map_img,
            aspect = hmax.get_aspect(),
            extent = hmax.get_xlim() + hmax.get_ylim(),
            zorder = 1)
        
        # Grabbing generated map from IO buffer and rendering it on html site
        canvas = FigureCanvas(fig)
        img = io.BytesIO()
        plt.savefig(img)
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        source = 'data:image/png;base64,{}'.format(plot_url)
        return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)
