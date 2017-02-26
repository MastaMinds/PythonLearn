# Beginning of the code
print "Hello World, This is my first Python program"

# Your name
name=raw_input("Please Type in your name: ")
print "Welcome to Python "+name

# European Floor 
EUf=raw_input("Europe Floor? ")
USf=int(EUf)+1
print "US Floor",USf

# Pay according to work hours and rate
hr=raw_input("Enter hours ")
rate=raw_input("Enter rate ")
try:
    hours=float(hr)
    ratepay=float(rate)
except:
    hours=-1
    ratepay=-1

if hours>0:
    print("For "+hr+" Hours, with a rate of "+rate+", The pay is:")
    if hours>40:
        pay=40*ratepay+(hours-40)*(1.5*ratepay)
        print pay
    else:
        pay=hours*ratepay
        print pay
else:
    print("Please enter numeric values")
    quit()