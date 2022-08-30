from flask import Flask,request,jsonify
import pickle
import numpy as np


model = pickle.load(open('model.pkl','rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return "Hello World"

@app.route('/predict', methods = ['POST'])
def predict():
    Q1 = request.form.get('Q1')
    Q2 = request.form.get('Q2')
    Q3 = request.form.get('Q3')
    Q4 = request.form.get('Q4')
    Q5 = request.form.get('Q5')
    Q6 = request.form.get('Q6')
    Q7 = request.form.get('Q7')
    Q8 = request.form.get('Q8')
    Q9 = request.form.get('Q9')
    Q10 = request.form.get('Q10')
    Q11 = request.form.get('Q11')
    Q12 = request.form.get('Q12')
    Q13 = request.form.get('Q13')
    Q14 = request.form.get('Q14')
    Q15 = request.form.get('Q15')
    Q16 = request.form.get('Q16')
    Q17 = request.form.get('Q17')
    Q18 = request.form.get('Q18')
    Q19 = request.form.get('Q19')
    Q20 = request.form.get('Q20')
    Q21 = request.form.get('Q21')

    result = {'Q1':Q1,'Q2':Q2,'Q3':Q3,'Q4':Q4,'Q5':Q5,'Q6':Q6,'Q7':Q7,
              'Q8':Q8,'Q9':Q9,'Q10':Q10,'Q11':Q11,'Q12':Q12,'Q13':Q13,
              'Q14':Q14,'Q15':Q15,'Q16':Q16,'Q17':Q17,'Q18':Q18,
              'Q19':Q19,'Q20':Q20,'Q21':Q21}

    input_query = np.array([[Q1,Q2,Q3,Q4,Q5,Q6,Q7,Q8,Q9,Q10,Q11,Q12,Q13,
                             Q14,Q15,Q16,Q17,Q18,Q19,Q20,Q21]])

    result = model.predict(input_query)[0]

    #if result == 1:
        #return jsonify({'Result:':str(result)})




    return jsonify({'Result':str(result)})

if __name__ == '__main__':
    app.run(debug = True)