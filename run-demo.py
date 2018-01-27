from flask import Flask, render_template, redirect, request, jsonify
from squad.demo_prepro import prepro
from basic.demo_cli import Demo
from IPython import embed
from my_question import my_question
import json

app = Flask(__name__)
#shared = json.load(open("data/squad/shared_test.json", "r"))
_data = json.load(open("data/squad/dev-v1.1.json", "r"))['data']
data = []
titles = []
for d in _data:
    titles.append(d['title'])
    data.append([])
    for par in d['paragraphs']:
        context = par["context"]
        questions = [q['question'] for q in par['qas']]
        data[-1].append((context, questions))

contextss = [""]
context_questions = [[]]
for i in range(len(data)):
    j = 24 if i==0 else 0
    contextss.append(data[i][j][0])
    context_questions.append(my_question.get(i, []) + data[i][j][1])

titles = ["Write own paragraph"] + ["[%s] %s" % (str(i).zfill(2), title) for i, title in enumerate(titles)]

demo = Demo()

def getTitle(ai):
    return titles[ai]

def getPara(rxi):
    return contextss[rxi[0]][rxi[1]]

def getAnswer(paragraph, question):
    pq_prepro = prepro(paragraph, question)
    if len(pq_prepro['x'])>1000:
        return "[Error] Sorry, the number of words in paragraph cannot be more than 1000." 
    if len(pq_prepro['q'])>100:
        return "[Error] Sorry, the number of words in question cannot be more than 100."
    return demo.run(pq_prepro)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    return jsonify(result={"titles" : titles, "contextss" : contextss, "context_questions" : context_questions})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraph = request.args.get('paragraph')
    question = request.args.get('question')
    answer = getAnswer(paragraph, question)
    print (question, answer)
    return jsonify(result=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1995, threaded=True)
