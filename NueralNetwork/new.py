import random
import os
import math

def matrixmult (A, B):
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    if cols_A != rows_B:
      print ("Cannot multiply the two matrices. Incorrect dimensions.")
      print("Col:", cols_A," Row:",rows_B)
      return
    
    C = [[0 for row in range(cols_B)] for col in range(rows_A)]
    
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                C[i][j] += A[i][k] * B[k][j]
    return C

def activation(x):
    return 1.0/(1+2.71**(-1*x))

def outputerror(x):
    return x*(1-x)*(output[0]-x)

def hiddenerror(x,err,nw):
    return x*(1-x)*(err*nw)

def weightupdate(prev,eta,err,out):
    return prev+(eta*err*out)

def prep_data_set(file):
     dataset=[]
     with open(file) as dataFile:
          for line in dataFile:
               line = line.split()
               for i in range(len(line)):
                    line[i]=float(line[i].strip("',"))
               dataset.append(list(line[0:]))
          return dataset
def prep_target(file):
     target=[]
     with open(file) as f:
          for line in f:
               target.append(int(line)-1)
          return target
def split_data_set(data):
     train=random.sample(data,int(len(data)*0.80))
     test=random.sample(data,int(len(data)*0.20))
     return (train,test)
     
target=prep_target("Target.txt")
#print (target)

dataset=prep_data_set("Dataset.txt")

for k in range(len(dataset)):
    dataset[k].append(target[k])

Train,Test=split_data_set(dataset)
#print (len(Train))

inptest=Test
inptest=[[[y] for y in x[0:61]]for x in inptest]

inpt=Train
inpt=[[[y] for y in x[0:61]]for x in inpt]
#print ((inpt[0]))

#inp =(dataset[0])
#inp = [[x] for x in inp]
#print (inp)

no_hid_layer=1
no_of_node=5

hid_weight=[[0.5 for y in range(61)] for x in range(no_of_node)]
#print (len(hid_weight[0]))
out_weight=[[0.5 for x in range(no_of_node)]]
#print (out_weight)
output=target

for x in range(100):
     Terr=0
     for i in range(len(inpt)): 
         hidin=matrixmult(hid_weight,inpt[i])
         #print (hidin)
         hidout=[[activation(x[0])] for x in hidin]
         #print (hidout)

         outin=matrixmult(out_weight,hidout)
         #print (outin)
         outout=activation(outin[0][0])
         #print outout
         #print Train[i][61]
         err = abs(Train[i][61]-outout)
         outerr=outputerror(err)
         out_weight=[[weightupdate(x,1,outerr,outout) for x in out_weight[0]]]
         #print (out_weight)

         hiderr=[]
         for i in range(len(hid_weight)):
             hiderr.append(hiddenerror(hidout[i][0],outerr,out_weight[0][i]))

         #print (hiderr)

         for i in range(len(hid_weight)):
             for j in range(len(hid_weight[0])):
                 hid_weight[i][j]=weightupdate(hid_weight[i][j],1,hiderr[i],hidout[i][0])
         #print (hid_weight)
     #print (Terr)
c=0
for i in range(len(inptest)): 
         hidin=matrixmult(hid_weight,inptest[i])
         #print (hidin)
         hidout=[[activation(x[0])] for x in hidin]
         #print (hidout)

         outin=matrixmult(out_weight,hidout)
         #print (outin)
         outout=activation(outin[0][0])
         #print "program",outout
         #print "actual",Train[i][61]
         #print "abs",abs(outout-Train[i][61])
         if(abs(outout-Train[i][61])<0.10):
             c=c+1

print "correctness: ",c*100.0/len(inptest),"%"
