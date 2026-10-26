import random,time,os,json,math
import cupy as cp
datei="DATA(06).txt"    #learn from
save="06_0.json"    #save too
dim=64      #for new training
hl=96       #for new training
context=10   #for new training
lr=0.001    #for new training
batchsize=16   #for new training
debug=False
slowmode=1
temperature=0.4

maxamount=1000 #for all training
saveafter=1 #for all training

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
    #emb,w1,w2,b1,b2
    return data

def savejson(file):
    global context,voka,emb,w1,w2,b1,b2,dim,hl,lr,words,epochss,batchsize
    print("SAVED")
    emb2={}
    for i in emb:
        emb2[i]=emb[i].tolist()
    data={
    "version":[1.1,1.2],
    "epochs":epochss,
    "context":context,
    "dim":dim,
    "hl":hl,
    "lr":lr,
    "batchsize":batchsize,
    "words":words,
    "voka":voka,
    "emb":emb2,
    "w1":w1.tolist(),
    "w2":w2.tolist(),
    "b1":b1.tolist(),
    "b2":b2.tolist()
    }
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    with open(path,"w") as f:
        json.dump(data,f)

def nodupe(text):
    return list(dict.fromkeys(text))                         #optimised

def nospace(text):
    return [i for i in text if i and i != " "]     #optimised

def softmax(x,temp=1):
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

def calcerr(w1,w2,b1,b2,inp,answer,dim):
    layer1=cp.maximum(0,w1@inp+b1)

    tmp=w2@layer1+b2
    layer2=softmax(tmp)

    outerr=answer-layer2

    hiddenerr=w2.T@outerr
    hiddenerr*=(layer1>0)

    inerr=w1.T@hiddenerr
    target=int(cp.argmax(answer))          #what the answer is(one hot)
    err=float(-cp.log(layer2[target]+1e-10))   # the error to that answer ( not mse anymore)

    gw1=cp.outer(hiddenerr, inp)
    gw2=cp.outer(outerr,layer1)
    gb1=hiddenerr
    gb2=outerr
    gemb=inerr.reshape(context, dim)
    return err,gw1,gw2,gb1,gb2,gemb


def update(w1,w2,b1,b2,gw1,gw2,gb1,gb2,gembw,count):
    w2+=lr*gw2/count

    b2+=lr*gb2/count

    w1+=lr*gw1/count

    b1+=lr*gb1/count
    for i in gembw:
        emb[i]+=lr*gembw[i]/count

def makegrad(w1,w2,b1,b2):
    gw1=cp.zeros_like(w1)
    gb1=cp.zeros_like(b1)
    gw2=cp.zeros_like(w2)
    gb2=cp.zeros_like(b2)
    return gw1,gw2,gb1,gb2

def train(w1,w2,b1,b2,maxamount,words,voka,emb,context,batchsize,dim):
    global epochss
    lup={w:i for i,w in enumerate(voka)}
    for i1 in range(maxamount):
        st=time.time()
        count=0
        tloss=0
        sample=0
        gw1,gw2,gb1,gb2=makegrad(w1,w2,b1,b2)
        gembw={}
        for contexxt in range(context):
            rc=context-contexxt
            if slowmode:
                print(contexxt)
            for w in range(len(words)-context-contexxt):
                answer=cp.zeros(len(voka),dtype=cp.float32)
                answer[lup[words[w+rc]]]=1
                wused=["$PAD$"]*contexxt+words[w:w+rc]#all the words used
                inp=cp.concatenate([emb[c] for c in wused])#all the emb
                tmp=calcerr(w1,w2,b1,b2,inp,answer,dim)
                gw1+=tmp[1]
                gw2+=tmp[2]
                gb1+=tmp[3]
                gb2+=tmp[4]
                for i,e in enumerate(wused):
                    if e not in gembw:
                        gembw[e]=cp.zeros(dim,dtype=cp.float32)
                    gembw[e]+=tmp[5][i]
                count+=1
                if count==batchsize:
                    update(w1,w2,b1,b2,gw1,gw2,gb1,gb2,gembw,count)
                    gw1.fill(0)
                    gw2.fill(0)
                    gb1.fill(0)
                    gb2.fill(0)
                    gembw={}
                    count=0
                tloss+=tmp[0]
                sample+=1
        if count>0:
            update(w1,w2,b1,b2,gw1,gw2,gb1,gb2,gembw,count)
        if i1%saveafter==0 and i1!=0:
            savejson(save)
        aloss=tloss/sample
        t=round((maxamount-(i1+1))*(time.time()-st))
        epochss+=1
        t=str(t//3600)+"h "+str(t%3600//60)+"min "+str(t%3600%60)+"sec"
        print("epoch",epochss+1,"time running:",t,"loss:",round(aloss,8))
        if aloss!=aloss:            #error by nan
            return "ERROR"
    savejson(save)

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
    a=train(w1,w2,b1,b2,maxamount,words,voka,emb,context,batchsize,dim)
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
    if "version" in tt and 1.2 in tt["version"]:
        epochss,context,voka,emb,w1,w2,b1,b2,words,dim,hl,lr,batchsize=tt["epochs"],tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["b1"],tt["b2"],tt["words"],tt["dim"],tt["hl"],tt["lr"],tt["batchsize"]
        a=train(w1,w2,b1,b2,maxamount,words,voka,emb,context,batchsize,dim)
        if a!="ERROR":
            savejson(save)
        else:
            print("ERROR")
    else:
        print("WRONG AI VERSION TO LOAD :[ try a newer format")
#batches