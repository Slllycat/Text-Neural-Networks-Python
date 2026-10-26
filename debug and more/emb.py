import json,os,random
import tkinter as tk
#THIS IS FOR VISUALIZING DIMENSIONS :D
file="01_1.json"
n=1
times=75
scale=1
press=0.2
show=[]

#window setup
window = tk.Tk()
window.title("BEST FUTURE WIFE EVER :D :P :O :I")
window.geometry(str(500*scale) + "x" + str(500*scale))
canvas = tk.Canvas(window, width=(500*scale), height=(500*scale), bg="white")
canvas.pack()

def loadjson(file):
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, file)
    with open(path,"r") as f:
        data=json.load(f)
    return data

data=loadjson(file)
emb=data["emb"]
emb=[(emb[i],i) for i in emb]
print(data["voka"])
print("\n Dimension lenght: ",data["dim"])
if n>data["dim"] or n<0:
    print("VALUES TO FAR!!!")
    exit("n error")

def move(event):
    global n
    if event.keysym=="plus":
        n+=1
    elif event.keysym=="minus":
        n-=1


window.bind("<Key>", move)


def render():
    global emb,n
    v=[(name,i[n]) for i,name in emb]
    a=[((i-len(emb)//2)//press) for i in range(len(emb))]
    canvas.delete("all")
    canvas.create_text(40*scale,20*scale,text="dim:"+str(n),fill="black",font=("Arial",10*scale))
    for i in range(len(v)):
        if v[i][0] in show or show==[]:
            value=v[i][1]
            x=(value*times*scale)+(250*scale)
            y=(a[i]*scale)+(250*scale)
            canvas.create_rectangle(x,y,x+2,y+2)
            canvas.create_text(x,y-10,text=v[i][0],fill="black",font=("Arial",7*scale))
    window.after(100,render)


render()
window.mainloop()