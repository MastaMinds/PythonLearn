def Welcome():
    print "Welcome to Python"

def GoodBye():
    print "Arrividerci!"

def greet(lang):
    if lang=="eng":
        ans="Hi"
    elif lang=="fr":
        ans="Bonjour"
    elif lang=="it":
        ans="Ciao"
    else:
        ans="Hola"
    
    return ans
    
def mult(a,b):
    c=a*b
    return c

Welcome()
print "Function test"
GoodBye()
print mult(10,20)

big=max("Hello world")
print big

name=raw_input("Input your first name: ")
lg=raw_input("Input the language code: eng,fr,it,es: ")
ans=greet(lg)
print ans,name+"!"