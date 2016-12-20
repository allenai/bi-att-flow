from flask import Flask, render_template, redirect, request, jsonify
from squad.demo_prepro import prepro
from basic.demo_cli import Demo
import json

app = Flask(__name__)
shared = json.load(open("data/squad/shared_test.json", "r"))
contextss = shared["contextss"]
titles = shared["titles"]
context_questions = shared["context_questions"]

demo = Demo()

def getTitle(ai):
    return titles[ai]

def getPara(rxi):
    return contextss[rxi[0]][rxi[1]]

def getAnswer(rxi, question):
    q_prepro = prepro(rxi, question)
    return demo.run(q_prepro)

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/select', methods=['GET', 'POST'])
def select():
    #paragraph_id = request.args.get('paragraph_id', type=int)
    #rxi = [paragraph_id, 0]
    #paragraph = getPara(rxi)
    #return jsonify(result=paragraph)
    return jsonify(result={"titles" : titles, "contextss" : contextss, "context_questions" : context_questions})

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    paragraph_id = request.args.get('paragraph_id', type=int)
    question = request.args.get('question')
    if paragraph_id == 0: rxi = [paragraph_id, 1]
    else: rxi = [paragraph_id, 0]
    answer = getAnswer(rxi, question)
    return jsonify(result=answer)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="1995", threaded=True)
