class Employee:
    emp_count=0
    #instances (special method(init))
    def __init__(self,name,salary):
        self.name=name
        self.salary=salary
        Employee.emp_count+=1
    #Methods
    def display_employee(self):
        print("name= %s, salary= %s"%(self.name,self.salary))
    
    def give_raise(self,percent):
        self.salary+=self.salary*percent
    
    def display_count(self):
        print("count=%d"%Employee.emp_count)

emp=Employee("Sam",1000)
emp.display_employee()
emp.give_raise(0.1)
emp.display_employee()
emp2=Employee("Sarah",2000)
emp2.display_employee()
emp.display_count()

emp.name="Ahmed"
emp.display_employee()