from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and features list
with open('app/model/titanic_model.pkl', 'rb') as f:
    model, features = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Get inputs from form
            pclass = int(request.form['pclass'])
            sex = request.form['sex']
            age = float(request.form['age'])
            sibsp = int(request.form['sibsp'])
            parch = int(request.form['parch'])
            fare = float(request.form['fare'])
            embarked = request.form['embarked']

            # Map sex and embarked like training
            sex_num = 0 if sex.lower() == 'male' else 1
            embarked_map = {'S': 0, 'C': 1, 'Q': 2}
            embarked_num = embarked_map.get(embarked.upper(), 0)

            # Create feature array in correct order
            features_input = np.array([[pclass, sex_num, age, sibsp, parch, fare, embarked_num]])

            pred = model.predict(features_input)[0]
            prediction = 'Survived' if pred == 1 else 'Not Survived'
        except Exception as e:
            prediction = f"Error in input: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
