import random,time,os,json,math
import cupy as cp
datei="DATA(03).txt"    #learn from
save="test.json"    #save too
dim=32      #for new training
hl=32       #for new training
context=10   #for new training
lr=0.001    #for new training
debug=False
slowmode=1
temperature=0.4

maxamount=1000 #for all training
saveafter=3 #for all training

def loadtxt(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    thefile = open(path, "r")
    all_lines = thefile.readlines()
    thefile.close()
    text=""
    for i in all_lines:
        text+=i
    return text

def loadjson(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    with open(path,"r") as f:
        data=json.load(f)
    for i in data["emb"]:
        data["emb"][i]=cp.array(data["emb"][i])
    data["w1"]=cp.array(data["w1"])
    data["w2"]=cp.array(data["w2"])
    data["b1"]=cp.array(data["b1"])
    data["b2"]=cp.array(data["b2"])
    emb,w1,w2,b1,b2
    return data

def savejson(file):
    global context,voka,emb,w1,w2,b1,b2,dim,hl,lr,words,epochss
    print("SAVED")
    emb2={}
    for i in emb:
        emb2[i]=emb[i].tolist()
    data={
    "version":[1.1],
    "epochs":epochss,
    "context":context,
    "words":words,
    "voka":voka,
    "emb":emb2,
    "w1":w1.tolist(),
    "w2":w2.tolist(),
    "b1":b1.tolist(),
    "b2":b2.tolist(),
    "dim":dim,
    "hl":hl,
    "lr":lr
    }
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    with open(path,"w") as f:
        json.dump(data,f)

def nodupe(text):
    return list(dict.fromkeys(text))                         #optimised

def nospace(text):
    return [i for i in text if i and i != " "]     #optimised

def softmax(x, temp=1):
    x=x-cp.max(x)
    e=cp.exp(x/temp)
    return e/cp.sum(e)

def neuron(weights,inputs,b):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return sum(temp)+b

def embeding(text,dim):
    emb={}
    emb["$PAD$"]=cp.zeros(dim,dtype=cp.float32)
    tmp=cp.random.uniform(-1,1,(len(text),dim),dtype=cp.float32)
    for i in zip(tmp,text):
        emb[i[1]]=i[0]
    return emb

def makeweights(leni,lenm,leno):
    w1=cp.random.uniform(-0.5,0.5,(lenm,leni),dtype=cp.float32)
    w2=cp.random.uniform(-0.5,0.5,(leno,lenm),dtype=cp.float32)
    b1=cp.zeros(lenm, dtype=cp.float32)
    b2=cp.zeros(leno, dtype=cp.float32)
    return w1,w2,b1,b2

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

def trainonce(w1,w2,b1,b2,inp,answer,wused):
    layer1=cp.maximum(0,w1@inp+b1)

    tmp=w2@layer1+b2
    layer2=softmax(tmp)

    outerr=answer-layer2

    hiddenerr=w2.T@outerr
    hiddenerr*=(layer1>0)

    inerr=w1.T @ hiddenerr

    w2+=lr*cp.outer(outerr,layer1)

    b2+=lr*outerr

    w1+=lr*cp.outer(hiddenerr, inp)

    b1+=lr*hiddenerr

    err_emb = inerr.reshape(context, dim)
    for c, w in enumerate(wused):
        emb[w] += lr * err_emb[c]

    target=int(cp.argmax(answer))          #what the answer is(one hot)
    error=float(-cp.log(layer2[target]+1e-10))   # the error to that answer ( not mse anymore)
    return error


def train(w1,w2,b1,b2,maxamount,words,voka,emb,context):
    global epochss
    lup={w:i for i,w in enumerate(voka)}
    for i in range(maxamount):
        st=time.time()
        tloss=0
        sample=0
        for contexxt in range(context):
            rc=context-contexxt
            if slowmode:print(contexxt)
            for w in range(len(words)-context-contexxt):
                answer=cp.zeros(len(voka),dtype=cp.float32)
                answer[lup[words[w+rc]]]=1
                wused=["$PAD$"]*contexxt+words[w:w+rc]#all the words used
                inp=cp.concatenate([emb[c] for c in wused])#all the emb
                errr=trainonce(w1,w2,b1,b2,inp,answer,wused)
                tloss+=errr
                sample+=1
        if i%saveafter==0 and i!=0:
            savejson(save)
        aloss=tloss/sample
        t=round((maxamount-(i+1))*(time.time()-st))
        epochss+=1
        t=str(t//3600)+"h "+str(t%3600//60)+"min "+str(t%3600%60)+"sec"
        print("epoch",epochss+1,"time running:",t,"loss:",round(aloss,8))
        if aloss!=aloss:            #error by nan
            return "ERROR"

def answer(w1,w2,b1,b2,inp,context):
    inp,finp=[closeword(i) for i in (inp).split(" ")[-context:]],[]
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
        ww=[(voka[i]+" "+str(round(p*100,4))+"%") for i,p in enumerate(probs) if p>0.0001]
        print(inp,":\n","\n".join(ww))

    for i,p in enumerate(probs):
        s += p
        if r <= s:
            return voka[i]

a=input("TRAIN FROM NEW/USE/TAIN WITH FILE[1/2/3]:")
if a=="1":
    words=loadtxt(datei).replace("\n","").replace("."," . ").split(" ")
    words=nospace(words)
    voka=nodupe(words)
    emb=embeding(voka,dim)
    w1,w2,b1,b2=makeweights(dim*context,hl,len(voka))
    epochss=0
    print(datei,save,sep="\n")
    print("Vokalenght:",len(voka))
    a=train(w1,w2,b1,b2,maxamount,words,voka,emb,context)
    if a!="ERROR":
        savejson(save)
elif a=="2":
    tt=loadjson(save)
    if "version" in tt and 1.1 in tt["version"]:
        context,voka,emb,w1,w2,b1,b2=tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["b1"],tt["b2"]
        while True:
            inp=input("TEXT:")
            w,t,i="","",0
            while t!="." and i<11:
                t=answer(w1,w2,b1,b2,inp+w,context)
                w+=" "+t
                i+=1
            print(w)
    else:
        print("WRONG AI VERSION TO LOAD :[ try a newer format")
elif a=="3":
    tt=loadjson(save)
    if "version" in tt and 1.1 in tt["version"]:
        epochss,context,voka,emb,w1,w2,b1,b2,words,dim,hl,lr=tt["epochs"],tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["b1"],tt["b2"],tt["words"],tt["dim"],tt["hl"],tt["lr"]
        a=train(w1,w2,b1,b2,maxamount,words,voka,emb,context)
        if a!="ERROR":
            savejson(save)
        else:
            print("ERROR")
    else:
        print("WRONG AI VERSION TO LOAD :[ try a newer format")
#everything reworked to work with gpu :O