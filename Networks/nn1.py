import random,time,os
amount=500
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
text=loadtext("DATA.txt")
text=text.split()

def wordtotoken(text,w,token):
    if w in text:
        duplist=noduplicat(text)
        return token[duplist.index(w)]
    else:
        exit("FUHHH")

def tokentoword(text,tokens):
    duplist = noduplicat(text)
    return duplist[tokens.index(max(tokens))]

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

def maketokenlist(text,tokens):
    duplist=noduplicat(text)
    temp=[]
    for i in text:
        temp.append(tokens[duplist.index(i)])
    return temp

def makeweights(tokens):
    wl1=8
    w1=[]
    w2=[]
    for _ in range(wl1):
        temp=[]
        for _ in range(len(tokens)):
            temp.append(random.uniform(-0.5,0.5))
        w1.append(temp)
    for _ in range(len(tokens)):
        temp=[]
        for _ in range(len(w1)):
            temp.append(random.uniform(-0.5,0.5))
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

def errormath(predict,correct):
    temp=[]
    for i in range(len(predict)):
        temp.append((correct[i]-predict[i])**2)
    return sum(temp)

def doprediction(inputs,w1,w2):
    temp=[]
    for i in range(len(w1)):
        temp.append(neuron(w1[i],inputs))
    temp2=[]
    for i in range(len(w2)):
        temp2.append(neuron(w2[i],temp))
    return temp2

token=maketokens(text)      #each word one token
w1,w2=makeweights(token)        #weights with the lenght of tokens
lsttokens=maketokenlist(text,token)         #a list with all words in order as tokens

def train(inputs,correct,w1,w2):
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

for times in range(amount):   # train many times
    starttime=time.time()
    for i in range(len(lsttokens)-1):
        inp = lsttokens[i]       # current word
        correct = lsttokens[i+1] # next word
        train(inp, correct, w1, w2)
    print(round((amount-(times+1))*(time.time()-starttime)),"sec")

for _ in range(100):
    question=input("FRAGE: ")
    if question in text:
        predict = doprediction(wordtotoken(text,question,token), w1, w2)
        question=tokentoword(text,predict)
        print(question,end=" ")
#1 context 1 word at a time :P
