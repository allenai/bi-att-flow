from squad.demo_prepro import prepro
from basic.demo_cli import Demo
import json

demo = Demo()
contextss = json.load(open("data/squad/shared_test.json", 'r'))['contextss']
print (len(contextss))
for i in range(30):
    print (len(contextss[i]))

def getPara(rxi):
    return contextss[rxi[0]][rxi[1]]

def getAnswer(rxi, question):
    print ("START")
    q_prepro = prepro(rxi, question)
    return demo.run(q_prepro)


print (getPara([0, 0]))
print (getAnswer([0, 0], "What 2015 NFL team one the AFC playoff?"))
