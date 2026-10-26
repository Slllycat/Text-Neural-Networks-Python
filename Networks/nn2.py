import random,time,os
datei="DATASHORT.txt"
amount=500
hl=8
context=4
lr=0.01

def loadtext(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    thefile = open(path, "r")
    all_lines = thefile.readlines()
    thefile.close()
    text=""
    for i in all_lines:
        text+=i
    return text

def tokentoword(text,tokens):
    text=nodupe(text)
    return text[tokens.index(max(tokens))]


def nodupe(text):
    tmp=[]
    for i in text:
        if not i in tmp:
            tmp.append(i)
    return tmp

def maketokens(text):
    text=nodupe(text)
    tmp=[]
    token={}
    for _ in range(len(text)):
        tmp.append(0)
    for i,e in enumerate(text):
        tmp2=tmp[:]
        tmp2[i]=1
        token[e]=tmp2
    return token

def makeweights(leni,lenm,leno):
    w1=[]
    w2=[]
    for _ in range(lenm):
        temp=[]
        for _ in range(leni):
            temp.append(random.uniform(-0.5,0.5))
        w1.append(temp)
    for _ in range(leno):
        temp=[]
        for _ in range(lenm):
            temp.append(random.uniform(-0.5,0.5))
        w2.append(temp)
    return w1,w2

def activation(n):
    if n>=0:
        return n
    else:
        return 0

def randomchoice(n):
    tmp=[]
    tmp.append(n.index(max(n)))
    for i,e in enumerate(n):
        for _ in range(round(e*100)):
            tmp.append(i)
    i=n.index(n[random.choice(tmp)])
    tmp=[]
    for _ in n:
        tmp.append(0)
    tmp[i]=1
    return tmp

def neuron(weights,inputs):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return activation(sum(temp))

def doprediction(inputs,w1,w2):
    temp=[]
    for i in range(len(w1)):
        temp.append(neuron(w1[i],inputs))
    temp2=[]
    for i in range(len(w2)):
        temp2.append(neuron(w2[i],temp))
    return randomchoice(temp2)
a=1
def trainonce(inputs,correct,w1,w2):
    hidden=[]
    for i in range(len(w1)):
        hidden.append(neuron(w1[i],inputs))
    output=[]
    for i in range(len(w2)):
        output.append(neuron(w2[i],hidden))
    output_error=[]
    for i in range(len(output)):
        output_error.append(correct[i]-output[i])
    for j in range(len(w2)):
        for i in range(len(w2[j])):
            w2[j][i]+=lr*output_error[j]*hidden[i]
    hidden_error=[0]*len(hidden)
    for i in range(len(hidden)):
        for j in range(len(output_error)):
            hidden_error[i]+=output_error[j]*w2[j][i]
    for i in range(len(w1)):
        for k in range(len(w1[i])):
            w1[i][k]+=lr*hidden_error[i]*inputs[k]

def train(amount,text,w1,w2):
    for times in range(amount):
        starttime=time.time()
        for wordsss in range(context):
            for i in range(len(text)-(context-wordsss)):
                inp=[]
                for i2 in range((context-wordsss)):
                    inp=inp+token[text[i+i2]]
                tmp=[0]*len(token[text[0]])
                for _ in range(wordsss):
                    inp=inp+tmp
                correct = token[text[i+(context-wordsss)]]
                trainonce(inp,correct,w1,w2)
        t=round((amount-(times+1))*(time.time()-starttime))
        if t!=0:
            t=str(t//3600)+"h "+str(t%3600//60)+"min "+str(t%3600%60)+"sec"
            print("epoch",times,"time:",t)

print("STARTING")
text=loadtext(datei)
print("STEP 1/5")
text=text.split()
print("STEP 2/5")
token=maketokens(text)
print("STEP 3/5")
l=len(nodupe(text))
w1,w2=makeweights(l*context,hl,l)
print("STEP 4/5")
train(amount,text,w1,w2)
print("STEP 5/5")
while True:
    question=input("WORD: ")
    bleh=question.split()[-context:]
    fehler=0
    frage=[]
    for i in bleh:
        if i in token:
            frage+=token[i]
        else:
            fehler=1
    if fehler==0:
        try:
            word=""
            predict=""
            while predict!=".":
                predict=doprediction(frage,w1,w2)
                predict=tokentoword(text,predict)
                frage+=token[predict]
                if len(frage)>context:
                    frage=frage[len(token[predict]):]
                word=word+" "+predict
            print(word)
        except:
            print("SOMETHING WENT WRONG")
#context and sentences
