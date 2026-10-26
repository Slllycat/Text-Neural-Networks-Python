import random,time,os,json,math
datei="DATA.txt"    #learn from
save="data.json"    #save too
dim=8      #for new training
hl=16        #for new training
context=3   #for new training
lr=0.001    #for new training

maxamount=500 #for all training
saveafter=2656 #for all training

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
    with open(file,"r") as f:
        data=json.load(f)
    return data

def savejson(file):
    global context,voka,emb,w1,w2,dim,hl,lr,words
    data={
    "context":context,
    "words":words,
    "voka":voka,
    "emb":emb,
    "w1":w1,
    "w2":w2,
    "dim":dim,
    "hl":hl,
    "lr":lr
    }
    with open(file,"w") as f:
        json.dump(data, f)

def nodupe(text):
    tmp=[]
    for i in text:
        if (not i in tmp):
            tmp.append(i)
    return tmp

def nospace(text):
    tmp=[]
    for i in text:
        if i!="" and i!=" ":
            tmp.append(i)
    return tmp

def softmax(x):
    m=max(x)        #to prevent overflow :O
    a=[]
    for i in x:
        a.append(math.exp(i-m))
    s=sum(a)
    a2=[]
    for i in a:
        a2.append(i/s)
    return a2

def neuron(weights,inputs):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return sum(temp)

def embeding(text,dim):
    emb={}
    emb["$PAD$"]=[0]*dim
    for i in range(len(text)):
        tmp=[]
        for _ in range(dim):
            tmp.append(random.uniform(-1,1))
        emb[text[i]]=tmp
    return emb

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

def closeword(w:str):
    bw,b="",0
    for i in emb:
        x=0
        for i2 in range(min(len(i),len(w))):
            if i[i2]==w[i2]:
                x+=1
        if len(i)==len(w):
            x+=1
        if x>b:
            b=x
            bw=i
    return bw

def trainonce(w1,w2,inp,answer,wused):
    layer1=[]
    for i in range(len(w1)):
        layer1.append(activation(neuron(w1[i],inp)))#relu
    tmp=[]
    for i in range(len(w2)):
        tmp.append(neuron(w2[i],layer1))
    layer2=softmax(tmp)

    outerr=[]
    for i in range(len(layer2)):
        outerr.append(answer[i]-layer2[i])

    for i in range(len(w2)):            #for each multiplier
            for i2 in range(len(w2[i])):    #for each value
                w2[i][i2]+=lr*outerr[i]*layer1[i2]#change each value by the error and the value of the previos layer.

    hiddenerr=[0]*len(layer1)           # not 4*10 values only 4 each error for vaules are added
    for i in range(len(layer1)):
        for i2 in range(len(outerr)):
            hiddenerr[i]+=outerr[i2]*w2[i2][i]

    for i in range(len(w1)):            #for each multiplier
        if layer1[i]>0:
            for i2 in range(len(w1[i])):    #for each value
                w1[i][i2]+=lr*hiddenerr[i]*inp[i2]                         #change each value by the error and the value of the previos layer.

    inerr=[0]*len(inp)
    for i in range(len(inp)):
        for i2 in range(len(w1)):

            inerr[i]+=hiddenerr[i2]*w1[i2][i]       # the higher the neoron the higher the error for it
    for c in range(context):                        # for each word that is in the imput
        w=wused[c]                                  # the word right now
        for d in range(dim):                        #for each vaule(dim) in each word
            emb[w][d]+= lr*inerr[c*dim+d]           #change it from the lr and the error

    target=answer.index(1)          #what the answer is
    error=-math.log(layer2[target]+1e-10)   # the error to that answer ( not mse anymore)
    return error


def train(w1,w2,maxamount,words,voka,emb,context):
    for i in range(maxamount):
        st=time.time()
        tloss=0
        sample=0
        for contexxt in range(context):
            for w in range(len(words)-context-contexxt):
                answer=[0]*len(voka)
                answer[voka.index(words[w+context-contexxt])]=1
                inp=[]
                wused=[]
                for c in range(contexxt):
                    inp.extend(emb["$PAD$"])
                    wused.append("$PAD$")
                for c in range(context-contexxt):
                    www=words[w+c]
                    inp.extend(emb[www])
                    wused.append(www)
                tloss+=trainonce(w1,w2,inp,answer,wused)
                sample+=1
        if i%saveafter==0 and i!=0:
            savejson(save)
        aloss=tloss/sample
        t=round((maxamount-(i+1))*(time.time()-st))
        t=str(t//3600)+"h "+str(t%3600//60)+"min "+str(t%3600%60)+"sec"
        print("epoch",i+1,"time running:",t,"loss:",round(aloss,8))

def answer(w1,w2,inp,context):
    inp,finp=(inp).split(" ")[-context:],[]
    for i in inp:
        if not i in emb:
            return ""
    while len(inp)<context:
        inp=["$PAD$"]+inp
    for i in inp:
        finp.extend(emb[i])
    tmp=[]
    for i in range(len(w1)):
        tmp.append(activation(neuron(w1[i],finp))) #relu
    tmp2=[]
    for i in range(len(w2)):
        tmp2.append((neuron(w2[i],tmp)))
    return voka[tmp2.index(max(tmp2))]

a=input("TRAIN FROM NEW/USE/TAIN WITH FILE[1/2/3]:")
if a=="1":
    words=loadtxt(datei).replace("\n","").replace("."," . ").split(" ")
    words=nospace(words)
    voka=nodupe(words)
    print("Vokalenght:",len(voka))
    emb=embeding(voka,dim)
    w1,w2=makeweights(dim*context,hl,len(voka))
    train(w1,w2,maxamount,words,voka,emb,context)
    savejson(save)
elif a=="2":
    tt=loadjson(save)
    context,voka,emb,w1,w2=tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"]
    while True:
        inp=input("TEXT:")
        w,t,i="","",0
        while t!="." and i<11:
            t=answer(w1,w2,inp+w,context)
            w+=" "+t
            i+=1
        print(w)
elif a=="3":
    tt=loadjson(save)
    context,voka,emb,w1,w2,words=tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["words"]
    train(w1,w2,maxamount,words,voka,emb,context,words)
    savejson(save)
#saving
