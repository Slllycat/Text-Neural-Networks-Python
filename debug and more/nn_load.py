import random,json,math,os
#THIS IS JUST FOR LOADING :P
file="03_1.json"
temperature=1
debug=0
showm=2


def loadjson(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    with open(path,"r") as f:
        data=json.load(f)
    return data

def softmax(x,temp=0.4):
    m=max(x)        #to prevent overflow :O
    a=[]
    for i in x:
        a.append(math.exp((i-m)/temp))
    s=sum(a)
    a2=[]
    for i in a:
        a2.append(i/s)
    return a2

def neuron(weights,inputs,b):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return sum(temp)+b

def activation(n):
    if n>=0:
        return n
    else:
        return 0

def closeword(w:str):
    if w in emb:
        return w
    else:
        return "$PAD$"

def answer(w1,w2,b1,b2,inp,context):
    inp,finp=[closeword(i) for i in (inp).split(" ")[-context:]],[]
    if "$PAD$" in inp:
        return ""
    for i in inp:
        if not i in emb:
            return ""
    while len(inp)<context:
        inp=["$PAD$"]+inp
    for i in inp:
        finp.extend(emb[i])
    tmp=[]
    for i in range(len(w1)):
        tmp.append(activation(neuron(w1[i],finp,b1[i]))) #relu
    tmp2=[]
    for i in range(len(w2)):
        tmp2.append((neuron(w2[i],tmp,b2[i])))
    probs = softmax(tmp2,temperature)
    r = random.random()
    s = 0
    if debug:
        ww=[voka[i] + " " + str(round(p*100, showm)) + "%"for i, p in sorted(enumerate(probs), key=lambda x: x[1], reverse=True)if p > 1/10**showm]
        print(inp,":\n","\n".join(ww),"\n")

    for i,p in enumerate(probs):
        s += p
        if r <= s:
            return voka[i]

tt=loadjson(file)
if "version" in tt and 1.1 in tt["version"]:
    context,voka,emb,w1,w2,b1,b2=tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["b1"],tt["b2"]
    while True:
        inp=input("TEXT:")
        thefile = open("DATAINPUTS.txt", "a")
        thefile.writelines(inp+"\n")
        thefile.close()
        w,t,i="","",0
        while t!="." and i<11:
            t=answer(w1,w2,b1,b2,inp+w,context)
            w+=" "+t
            i+=1
        print(w)
else:
    print("WRONG AI VERSION TO LOAD :[ try a newer format")
