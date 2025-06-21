from flask import Flask, render_template, request
import pickle
import numpy as np
from flask import make_response
from xhtml2pdf import pisa
from io import BytesIO
from flask import send_file

app = Flask(__name__)

model = pickle.load(open("model/model.pkl", "rb"))

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/medical', methods=["POST"])
def medical():
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    gender = request.form['gender']
    return render_template("medical_form.html", name=name, email=email, age=age, gender=gender)

@app.route('/result', methods=["POST"])
def result():
    # Get personal info from hidden fields
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    gender = request.form['gender']

    # Get medical info
    features = [float(request.form[field]) for field in [
        'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
        'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]]
    
    # Add age and gender to features
    features.insert(0, float(age))
    features.insert(1, float(gender))  # 1 for male, 0 for female

    prediction = model.predict([features])[0]
    result = "High Risk of Heart Disease 💔" if prediction == 1 else "Low Risk of Heart Disease ❤️"

    return render_template("result.html",
                           name=name,
                           email=email,
                           age=age,
                           gender="Male" if gender == "1" else "Female",
                           features=request.form,
                           result=result)


@app.route("/download", methods=["POST"])
def download():
    name = request.form['name']
    email = request.form['email']
    age = request.form['age']
    gender = "Male" if request.form['gender'] == "1" else "Female"
    result = request.form['result']
    
    # Extract all medical inputs
    medical_fields = [
        'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ]
    medical_data = {field: request.form[field] for field in medical_fields}
    
    # Render HTML
    html = render_template("pdf_template.html", name=name, email=email, age=age,
                           gender=gender, result=result, features=medical_data)

    # Convert HTML to PDF
    pdf = BytesIO()
    pisa_status = pisa.CreatePDF(html, dest=pdf)

    pdf.seek(0)
    return send_file(pdf, as_attachment=True, download_name="heart_risk_report.pdf", mimetype='application/pdf')


if __name__ == "__main__":
    app.run(debug=True)
