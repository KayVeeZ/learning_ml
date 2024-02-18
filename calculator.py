# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:49:22 2023

@author: Kshitij Vashisth
"""
print("My First Calculator")
a= float(input("Enter First Number: "))
b= float(input("Enter Second Number: "))
c= int(input("Choose number for operation to be performed:\n1.Addition\n2.Subtraction\n3.Multiplication\n4.Division\n"))
if(c==1):
  print(a+b)
elif(c==2):
  print(a-b)
elif(c==3):
  print(a*b)
elif(c==4):
  print(a/b)
else:
  print("Invalid operation")