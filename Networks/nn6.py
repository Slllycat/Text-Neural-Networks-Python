import random,time,os,json,math
datei="DATA(04).txt"    #learn from
save="4_0.json"    #save too
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
    with open(file,"r") as f:
        data=json.load(f)
    return data

def savejson(file):
    global context,voka,emb,w1,w2,b1,b2,dim,hl,lr,words,epochss
    data={
    "version":[1.1],
    "epochs":epochss,
    "context":context,
    "words":words,
    "voka":voka,
    "emb":emb,
    "w1":w1,
    "w2":w2,
    "b1":b1,
    "b2":b2,
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

def softmax(x,temp=1):
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
    w1,w2,b1,b2=[],[],[],[]
    for _ in range(lenm):
        temp=[]
        for _ in range(leni):
            temp.append(random.uniform(-0.5,0.5))
        w1.append(temp)
    for _ in range(lenm):
        b1.append(0)
    for _ in range(leno):
        b2.append(0)
        temp=[]
        for _ in range(lenm):
            temp.append(random.uniform(-0.5,0.5))
        w2.append(temp)
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
    layer1=[]
    for i in range(len(w1)):
        layer1.append(activation(neuron(w1[i],inp,b1[i])))#relu
    tmp=[]
    for i in range(len(w2)):
        tmp.append(neuron(w2[i],layer1,b2[i]))
    layer2=softmax(tmp)

    outerr=[]
    for i in range(len(layer2)):
        outerr.append(answer[i]-layer2[i])

    hiddenerr=[0]*len(layer1)           # not 4*10 values only 4 each error for vaules are added
    for i in range(len(layer1)):
        for i2 in range(len(outerr)):
            hiddenerr[i]+=outerr[i2]*w2[i2][i]

    for i in range(len(layer1)):    #making sure dead neurons stay dead
        if layer1[i] <= 0:
            hiddenerr[i]=0

    inerr=[0]*len(inp)
    for i in range(len(inp)):
        for i2 in range(len(w1)):
            inerr[i]+=hiddenerr[i2]*w1[i2][i]

    for i in range(len(w2)):            #for each multiplier
        for i2 in range(len(w2[i])):    #for each value
            w2[i][i2]+=lr*outerr[i]*layer1[i2]#change each value by the error and the value of the previos layer.

    for i in range(len(b2)):
        b2[i]+=lr*outerr[i]    #the error for each neoron

    for i in range(len(w1)):            #for each multiplier
        if layer1[i]>0:
            for i2 in range(len(w1[i])):    #for each value
                w1[i][i2]+=lr*hiddenerr[i]*inp[i2]                         #change each value by the error and the value of the previos layer.

    for i in range(len(b1)):
        if layer1[i]>0: #relu
            b1[i]+=lr *hiddenerr[i]
       # the higher the neoron the higher the error for it

    for c in range(context):                        # for each word that is in the imput
        w=wused[c]                                  # the word right now
        for d in range(dim):                        #for each vaule(dim) in each word
            emb[w][d]+= lr*inerr[c*dim+d]           #change it from the lr and the error

    target=answer.index(1)          #what the answer is
    error=-math.log(layer2[target]+1e-10)   # the error to that answer ( not mse anymore)
    return error


def train(w1,w2,b1,b2,maxamount,words,voka,emb,context):
    global epochss
    aloss2=9999999999999
    for i in range(maxamount):
        st=time.time()
        tloss=0
        sample=0
        for contexxt in range(context):
            if slowmode:print(contexxt)
            if debug:
                print("max w1",max([max(i) for i in w1]),min([min(i) for i in w1]))
                print("max w2",max([max(i) for i in w2]),min([min(i) for i in w2]))
                print("max emb",max([max(emb[i])for i in emb]),min([min(emb[i])for i in emb]))
            for w in range(len(words)-context-contexxt):
                answer=[0]*len(voka)
                answer[voka.index(words[w+context-contexxt])]=1
                inp=[]
                wused=[]
                for c in range(contexxt):
                    inp.extend(emb["$PAD$"])
                    wused.append("$PAD$")
                for c in range(context-contexxt):
                    www=words[w+c]#the current word
                    inp.extend(emb[www])#all the emb
                    wused.append(www)#all the words
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
        #if aloss2*1.05<aloss:      #error by nan
        #    return "ERROR"
        aloss2=aloss

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
        epochss,context,voka,emb,w1,w2,b1,b2,words=tt["epochs"],tt["context"],tt["voka"],tt["emb"],tt["w1"],tt["w2"],tt["b1"],tt["b2"],tt["words"]
        a=train(w1,w2,b1,b2,maxamount,words,voka,emb,context)
        if a!="ERROR":
            savejson(save)
        else:
            print("ERROR")
    else:
        print("WRONG AI VERSION TO LOAD :[ try a newer format")
#biases
