f=open("Git Commands.txt","r")
Command=f.read()
print Command
f.seek(20)
New=f.readline().rstrip()
New2=f.readline().rstrip()
New3=f.readline().rstrip()
New4=f.readlines()
print New,"\n",New2,"\n",New3
print New4
f.close()

f=open("sample.txt","w")
f.write("I am Ahmed\nI am an Aeronautical Engineer-polimi!")
f.close()

with open("sample2.txt","w") as f:
    f.write("This is the second python trial")
    
with open("In.txt") as f:
    with open("Out.txt","w") as out:
        for line in f:
            out.write(line)
