import numpy as np
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template, redirect
from joblib import load

# Load the model and the scaler
model = load_model("Tsla.h5")
scaler = load('scaler.joblib')

# Load the data
data_user = pd.read_csv("TSLA.csv")

graph = tf.get_default_graph()
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    # Get the username and password from the form
    username, password = request.form.values()

    # Connect to the database
    db = pymysql.connect("localhost", "root", '', "ddbb")
    cursor = db.cursor()

    # Check if the username already exists
    cursor.execute("SELECT user FROM user_register")
    usernames = [row[0] for row in cursor.fetchall()]
    if username in usernames:
        return render_template('register.html', text = "This Username is Already In Use.")

    # Insert the new user into the database
    sql = "INSERT INTO user_register(user, password) VALUES(%s,%s)"
    val = (username, password)
    try:
        cursor.execute(sql,val)
        db.commit()
    except:
        db.rollback()
    db.close()

    return render_template('register.html', text = "Successfully Registered.")

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/loggedin',methods=['POST'])
def loggedin():
    # Get the username and password from the form
    username, password = request.form.values()

    # Connect to the database
    db = pymysql.connect("localhost","root","","ddbb")
    cursor = db.cursor()

    # Fetch the password for the given username
    cursor.execute("SELECT password FROM user_register WHERE user = %s", (username,))
    result = cursor.fetchone()

    if result is not None and result[0] == password:
        return render_template('index1.html')
    else:
        return render_template('login.html', text="Use Proper Username and Password")

@app.route('/production')
def production():
    return render_template('index1.html')

@app.route('/production/predict',methods=['POST'])
def predict():
    date_required = request.form['prediction_date']
    num_days = 300  #number of days into the future want to predict
    
    # Get the most recent data
    recent_data = np.array(data_user['Close'])
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    
    # Creating a list to hold the predictions
    predictions = []
    
    # Loop for the number of days you want to predict
    for i in range(num_days):
        #Take the last 120 days of data
        seq = recent_data_scaled[-120:]
        
        # Reshape for input into the LSTM
        seq = np.reshape(seq, (1, 120, 1))
        
        # Make a prediction on the sequence with the LSTM
        with graph.as_default():
            predicted_price = model.predict(seq)
            
        # Append the predicted price to our list of predictions
        predictions.append(predicted_price[0, 0])
        
        # Append the predicted price to our recent data
        recent_data_scaled = np.concatenate((recent_data_scaled, predicted_price), axis=0)
        
        # Reshape the data for retraining the model
        x_train = np.reshape(recent_data_scaled[-120:], (1, 120, 1))
        y_train = np.array([recent_data_scaled[-1]])
        
        # Retrain the model with the new data
        model.fit(x_train, y_train, epochs=1, verbose=0)
    
    # Inverse transform the predictions to get them back on the original scale
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    return render_template('index1.html', prediction_text = 'The closing Value on {} will be {}'.format(date_required, predictions[0]))

@app.route('/admin')
def admin():
    return render_template('admin.html')

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        password = request.form['password']
        if password == 'adminLogin':
            return redirect('/admin')
        else:
            return 'Wrong Password'
    return render_template('admin_login.html')

@app.route('/admin/update', methods=['POST'])
def update_user_register():
    username, password = request.form.values()
    
    db = pymysql.connect("localhost", "root", '', "ddbb")
    cursor = db.cursor()
    sql = "UPDATE user_register SET password = %s WHERE user = %s"
    val = (password, username)
    
    try:
        cursor.execute(sql,val)
        db.commit()
    except:
        db.rollback()
    db.close()
    return redirect('http://localhost/phpmyadmin/')

if __name__ == "__main__":
    app.run(debug=True)
