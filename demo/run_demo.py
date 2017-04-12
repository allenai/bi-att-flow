from flask import Flask, render_template, redirect, request, jsonify
from basic.inference import Inference
import json

app = Flask(__name__)

#shared = json.load(open("data/squad/shared_demo.json", "r"))
#shared = json.load(open("data/squad/shared_specifiedby.json", "r"))
#data = json.load(open("data/squad/data_specifiedby.json", "r"))
shared = json.load(open("./../data/squad/shared_test.json", "r"))
data = json.load(open("./../data/squad/data_test.json", "r"))

context = [[x for x in xy] for xy in shared['p']]

article_titles = [
    "Super_Bowl_50",
    "Warsaw",
    "Normans",
    "Nikola_Tesla",
    "Computational_complexity_theory",
    "Teacher",
    "Martin_Luther",
    "Southern_California",
    "Sky_(United_Kingdom)",
    "Victoria_(Australia)",
    "Huguenot",
    "Steam_engine",
    "Oxygen",
    "1973_oil_crisis",
    "Apollo_program",
    "European_Union_law",
    "Amazon_rainforest",
    "Ctenophora",
    "Fresno,_California",
    "Packet_switching",
    "Black_Death",
    "Geology",
    "Newcastle_upon_Tyne",
    "Victoria_and_Albert_Museum",
    "American_Broadcasting_Company",
    "Genghis_Khan",
    "Pharmacy",
    "Immune_system",
    "Civil_disobedience",
    "Construction",
    "Private_school",
    "Harvard_University",
    "Jacksonville,_Florida",
    "Economic_inequality",
    "Doctor_Who",
    "University_of_Chicago",
    "Yuan_dynasty",
    "Kenya",
    "Intergovernmental_Panel_on_Climate_Change",
    "Chloroplast",
    "Prime_number",
    "Rhine",
    "Scottish_Parliament",
    "Islamism",
    "Imperialism",
    "United_Methodist_Church",
    "French_and_Indian_WarForce"
]
# Store questions in a similar data structure like contexts
context_questions = {}
question_index = 0
for article_idx, paragraph_idx in data['*p']:

    try:
        context_questions[article_idx]
    except KeyError:
        context_questions[article_idx] = {}

    try:
        context_questions[article_idx][paragraph_idx][0]
    except KeyError:
        context_questions[article_idx][paragraph_idx] = []

    context_questions[article_idx][paragraph_idx].append(data['q'][question_index])
    question_index += 1

inference = Inference()
# inference.data_ready()


@app.route('/')
def main():
    return render_template('index.html',
        contexts=json.dumps(context),
        context=context,
        titles=article_titles,
        num_pairs=len(article_titles),
        context_questions=json.dumps(context_questions)
    )

@app.route('/submit', methods=['POST'])
def submit():
    answers, scores = inference.predict(request.form['context'], request.form['question'])
    return jsonify(
        answers=answers,
        scores=[float(score) for score in scores]
    )

if __name__ == "__main__":
    from werkzeug.contrib.profiler import ProfilerMiddleware
    app.config['PROFILE'] = True
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, restrictions=[30])
    # app.run(port=5000, debug=True, threaded=True)
    app.run(host="0.0.0.0", port=1995, threaded=True, debug=True)
