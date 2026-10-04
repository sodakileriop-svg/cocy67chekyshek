from flask import Flask, render_template, request
from transformers import pipeline

pipe = pipeline("text-classification", model="blanchefort/rubert-base-cased-sentiment")

app = Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")


@app.route('/submit', methods=['POST'])
def submit():

    user_message = request.form.get("message", "")

    result = pipe(user_message)[0]
    label = result['label']

    return label


    # if user_message == "":
    #     return "ЧЁ МОЛЧИШЬ, ПИШИ ДАВАЙ!"
    # else:
    #     return f"Ты считаешь что Сталин {user_message}?"

if __name__ == "__main__":
    app.run()


# transformers, torch
