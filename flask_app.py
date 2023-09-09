# Import libraries
from flask import Flask, render_template, request
from sqlalchemy import create_engine
from urllib.parse import quote
import pandas as pd
import pickle
import joblib

imp_enc_scale = joblib.load('processed1')  # Imputation and Scaling pipeline
model = pickle.load(open('Clust_ins.pkl', 'rb')) # KMeans clustering model
model2 = pickle.load(open('db.pkl', 'rb'))
winsor = joblib.load('winsor')

def kmeans(data_new):
    clean1 = pd.DataFrame(imp_enc_scale.transform(data_new), 
                          columns = imp_enc_scale.get_feature_names_out())
    
    clean1[['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount']] = winsor.transform(clean1[['numerical__Customer Lifetime Value', 'numerical__Income',
       'numerical__Monthly Premium Auto', 'numerical__Months Since Last Claim',
       'numerical__Months Since Policy Inception', 'numerical__Number of Policies',
       'numerical__Total Claim Amount']])
   
    prediction = pd.DataFrame(model.predict(clean1), columns = ['Kmeans_clusters'])
    prediction1 = pd.DataFrame(model2.fit_predict(clean1), columns = ['db_scan_clusters'])
    
    final_data = pd.concat([prediction,prediction1 ,data_new], axis = 1)
    return(final_data)



app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        user = request.form['user']
        pw = request.form['password']
        db = request.form['databasename']
        engine = create_engine(f"mysql+pymysql://{user}:%s@localhost/{db}" % quote (f'{pw}'))
        try:

            data = pd.read_csv(f)
        except:
                try:
                    data = pd.read_excel(f)
                except:      
                    data = pd.DataFrame(f)
                    
                  
        # Drop the unwanted features
        ins_df = data.drop(["Customer",'Effective To Date', 'Location Code'], axis = 1)

        prediction = kmeans(ins_df)
        
        prediction.to_sql('ins_pred_kmeans', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = prediction.to_html(classes = 'table table-striped')
        
        return render_template("data.html", Y = f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #888a9e;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}")

if __name__=='__main__':
    app.run(debug = True)
