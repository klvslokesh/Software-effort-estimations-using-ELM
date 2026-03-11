import matplotlib
matplotlib.use("Agg")

from flask import Flask, render_template, request
from model import predict_effort, get_comparison_results
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder="templates")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    error = None

    if request.method == "POST":
        try:
            project_size = float(request.form["project_size"])
            team_size = float(request.form["team_size"])
            experience = float(request.form["experience"])

            # input validation
            if not (50 <= project_size <= 500):
                error = "Project size must be between 50 and 500"

            elif not (3 <= team_size <= 15):
                error = "Team size must be between 3 and 15"

            elif not (1 <= experience <= 10):
                error = "Experience must be between 1 and 10"

            else:
                prediction = predict_effort(project_size, team_size, experience)

        except:
            error = "Invalid input"

    return render_template("index.html", prediction=prediction, error=error)


@app.route("/comparison")
def comparison():

    results = get_comparison_results()

    models = list(results.keys())
    errors = list(results.values())

    plt.figure(figsize=(6,4))
    plt.bar(models, errors)

    plt.ylabel("MAE Error")
    plt.title("Algorithm Comparison")

    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)

    plot_url = base64.b64encode(img.getvalue()).decode()

    plt.close()

    return render_template("comparison.html", plot_url=plot_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=10000)
