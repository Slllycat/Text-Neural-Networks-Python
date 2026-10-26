import random
text="cats like eating fish and dogs like eating meat Humans like eating candy"
text=text.split()

def noduplicat(liste):
    temp=[]
    for i in liste:
        if not i in temp:
            temp.append(i)
    return temp

def maketokens(text):
    text=noduplicat(text)
    token=[]
    for i in range(len(text)):
        temp=[]
        for _ in range(len(text)):
            temp.append(0)
        temp[i]=1
        token.append(temp)
    return token

def makeweights(tokens):
    wl1=8
    w1=[]
    w2=[]
    for _ in range(wl1):
        temp=[]
        for _ in range(len(tokens)):
            temp.append(round(random.uniform(-1,1),2))
        w1.append(temp)
    for _ in range(len(tokens)):
        temp=[]
        for _ in range(len(w1)):
            temp.append(round(random.uniform(-1,1),2))
        w2.append(temp)
    return w1,w2

def activation(n):
    if n>=0:
        return n
    else:
        return 0

def neuron(weights,inputs):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return activation(sum(temp))

token=maketokens(text)
w1,w2=makeweights(token)
def doprediction(inputs,w1,w2):
    temp=[]
    for i in range(len(w1)):
        temp.append(neuron(w1[i],inputs))
    temp2=[]
    for i in range(len(w2)):
        temp2.append(neuron(w2[i],temp))
    print(temp2)
doprediction([1,0,0,0,0,0],w1,w2)
