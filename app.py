from flask import Flask, render_template, request
from model import predict_news

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    confidence = 0
    news_text = ''

    if request.method == "POST":
        news_text = request.form.get("news", "")
        if news_text.strip() != "":
            result, confidence = predict_news(news_text)

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        news_text=news_text
    )

if __name__ == "__main__":
    app.run(debug=True)