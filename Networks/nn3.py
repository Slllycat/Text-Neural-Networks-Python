import random,time,os
datei="DATALONG.txt"
dim=32
hl=128
context=6
lr=0.001
maxamount=500
def loadtext(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    thefile = open(path, "r")
    all_lines = thefile.readlines()
    thefile.close()
    text=""
    for i in all_lines:
        text+=i
    return text.replace("\n","").replace("."," . ")

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

def neuron(weights,inputs):
    temp=[]
    for i,e in enumerate(inputs):
        temp.append(e*weights[i])
    return activation(sum(temp))

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

def trainonce(w1,w2,inp,answer,wused):
    layer1=[]
    for i in range(len(w1)):
        layer1.append(neuron(w1[i],inp))
    layer2=[]
    for i in range(len(w2)):
        layer2.append(neuron(w2[i],layer1))

    outerr=[]
    for i in range(len(layer2)):
        outerr.append(answer[i]-layer2[i])  #this shit does: real - chance what it thinks

    for i in range(len(w2)):            #for each multiplier
        if layer2[i]>0:     #relu
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

    error=0
    for i in range(len(layer2)):
        error+=(answer[i]-layer2[i])**2
    error/=len(layer2)
    return error




def train(w1,w2,maxamount,words,voka,emb,context):
    st=time.time()
    for i in range(maxamount):
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
        aloss=tloss/sample
        t=round(time.time()-st)
        t=str(t//3600)+"h "+str(t%3600//60)+"min "+str(t%3600%60)+"sec"
        print("epoch",i+1,"time running:",t,"loss:",round(aloss,8))
        if aloss<0.001:
            return

def answer(w1,w2,inp,context):
    inp,finp=(inp).split(" ")[-context:],[]
    while len(inp)<context:
        inp=["$PAD$"]+inp
    for i in inp:
        finp.extend(emb[i])
    tmp=[]
    for i in range(len(w1)):
        tmp.append(neuron(w1[i],finp))
    tmp2=[]
    for i in range(len(w2)):
        tmp2.append((neuron(w2[i],tmp)))
    return voka[tmp2.index(max(tmp2))]

words=loadtext(datei).split(" ")
words=nospace(words)
voka=nodupe(words)
print("Vokalenght:",len(voka))
emb=embeding(voka,dim)
w1,w2=makeweights(dim*context,hl,len(voka))
train(w1,w2,maxamount,words,voka,emb,context)

while True:
    inp=input("TEXT:")
    w,t,i="","",0
    while t!="." and i<11:
        t=answer(w1,w2,inp+w,context)
        w+=" "+t
        i+=1
    print(w)
#embeding
