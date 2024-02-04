import numpy as np
import pandas as pd
import pymysql
pymysql.install_as_MySQLdb()
import warnings
warnings.filterwarnings("ignore")

username_list = []

import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, request, jsonify, render_template, redirect
from joblib import load

model = load_model("Apple.h5")
data_user = pd.read_csv("AAPL.csv")

graph = tf.get_default_graph()
app = Flask(__name__)

# Load the scaler
scaler = load('scaler.joblib')

updated_date = []
updated_close = []

@app.route('/')
def home():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def register():
    int_features2 = [str(x) for x in request.form.values()]
    
    r1 = int_features2[0]
    print(r1)
    
    r2 = int_features2[1]
    print(r2)

    username = int_features2[0]
    
    
# Open database connection
    db = pymysql.connect("localhost", "root", '', "ddbb")

# Prepare a cursor object using cursor() method
    cursor = db.cursor()
    cursor.execute("SELECT user FROM user_register")
    result1 = cursor.fetchall()
    
    for row1 in result1:
        print(row1)
        print(row1[0])
        username_list.append(str(row1[0]))
    
    print(username_list)
    if username in username_list:
        return render_template('register.html', text = "This Username is Already In Use.")
    else:
# Prepare SQL query to insert a record into the database
        sql = "INSERT INTO user_register(user, password) VALUES(%s,%s)"
        val = (r1, r2)
# Execute the SQL Command        
        try:
            cursor.execute(sql,val)
            db.commit()
        except:
            db.rollback()
        db.close()
        return render_template('register.html', text = "Successfully Registerd.")


@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/logedin',methods=['POST'])
def logedin():
    int_features3 = [str(x) for x in request.form.values()]
    logu = int_features3[0]
    passw = int_features3[1]

    # Open database connection
    db = pymysql.connect("localhost","root","","ddbb")

    # Prepare a cursor object using cursor() method
    cursor = db.cursor()

    # Fetch the password for the given username
    cursor.execute("SELECT password FROM user_register WHERE user = %s", (logu,))
    result = cursor.fetchone()

    if result is not None:
        stored_password = result[0]
        if stored_password == passw:
            return render_template('index1.html')
        
    return render_template('login.html', text="Use Proper Username and Password")


@app.route('/production')
def production():
    return render_template('index1.html')


@app.route('/production/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    date_required = request.form['prediction_date']
    num_days = 300  #number of days into the future want to predict
    
    # Get the most recent 120 days of closing prices
    recent_data = np.array(data_user['Close'][-120:])
    recent_data_scaled = scaler.transform(recent_data.reshape(-1, 1))
    
    # Creating a list to hold the predictions
    predictions = []
    
    
    # Loop for the number of days you want to predict
    for _ in range(num_days):
        #Take the last 120 days of data plus any new predictions
        seq = recent_data_scaled[-120:]
        
        # Reshape for input into the LSTM
        seq = np.reshape(seq, (1, 120, 1))
        
        # Make a prediction on the sequence with the LSTM
        with graph.as_default():
            predicted_price = model.predict(seq)
            
        # Append the predicted price to our list of predictions
        predictions.append(predicted_price[0, 0])
        
        # Append the predicted price to our sequence of recent data
        recent_data_scaled = np.concatenate((recent_data_scaled, predicted_price))

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
    int_features2 = [str(x) for x in request.form.values()]
    r1 = int_features2[0]
    r2 = int_features2[1]
    
    db = pymysql.connect("localhost", "root", '', "ddbb")
    cursor = db.cursor()
    sql = "UPDATE user_register SET password = %s WHERE user = %s"
    val = (r2, r1)
    
    try:
        cursor.execute(sql,val)
        db.commit()
    except:
        db.rollback()
    db.close()
    return redirect('http://localhost/phpmyadmin/')

if __name__ == "__main__":
    app.run(debug=True)