n=5
while n>0:
	print n
	n=n-1
print "Blastoff!"
print n

friends=["Bob3a","Dijaja","7aleeb"]
for friend in friends:
	print friend,"Al Salam Alaikom!"
print "Done!"

for i in range(5):
    print i

big=0
print "Initial value=",big
for num in [4,6,19,-4,41,9,94]:
    if num>big:
        big=num
    print big,num
print "Final value:",big

small=None
print "Initial value=",small
for num in [4,6,19,-4,41,9,94]:
    if small is None:
        small=num
    elif num<small:
        small=num
    print small,num
print "Final value:",small

found=False
for value in [9,41,12,3,74,15]:
    if value==3:
        found=True
        break
    print found,value

def sum_list(lister):
    sum=0
    for x in lister:
        sum+=x
    return sum

MyList=[1,2,3,4]
sum=sum_list(MyList)
print sum

fruit="Avocado"
print fruit[0:4]
print fruit[4:]
count=0
for letter in fruit:
    print letter
    if letter=="o":
        count=count+1
print count

    
num2=raw_input("Input a number:")
num2=int(num2)
mult=1
for i in range(num2):
    if i!=0:
        mult=mult*i
mult=mult*num2    
print "The factorial of",num2,"is",mult