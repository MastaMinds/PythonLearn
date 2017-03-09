# Try and except commands
Userip=raw_input("Enter a number: ")
try:
    ival=int(Userip)
except:
    ival=-1

if ival>0:
    print("Well done!")
else:
    print("Not a number..")
    
    

str1="Hello Ahmed"
try:
    a=int(str1)
except:
    a=-1
print "First",a

str2="123"
try:
    b=int(str2)
except:
    b=-1
print "Second",b

fruit="mango"
letter=fruit[2]
print letter