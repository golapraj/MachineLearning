import os
import random
import math

pos_dataset=[]
neg_dataset=[]


def write_object(object,filename):
    with open(filename, 'w' ) as W:
        for item in object:
            W.write(str(item)+ '\n' )

def read_review(dirn,filename):
    dataset=[]
    for file in os.listdir(dirn):
        with open(os.path.join(dirn, file)) as R:
            for line in R:
                line=line.split()
                #print lines
                dataset.append(list(line))

    write_object(dataset,filename)
    return (dataset)


def split_dataset(pos,neg):
    pos_len=len(pos_dataset)
    neg_len=len(neg_dataset)
    #print pos_len,neg_len

    pos_train=random.sample(pos,int(pos_len*0.80))
    neg_train=random.sample(neg,int(pos_len*0.80))
    pos_test=random.sample(pos,int(pos_len*0.20))
    neg_test=random.sample(neg,int(pos_len*0.20))

    return (pos_train,neg_train,pos_test,neg_test)

def build_vocabulary(pos_train,neg_train):
    vocabulary={}
    for sample in pos_train:
        for word in sample:
            word=word.lower().strip("'/\!@#$%&*+.:)(?-")
            if word == "":
                continue
            if vocabulary.__contains__(word):
                freq,T=vocabulary[word]
                freq=freq+1
                vocabulary[word]=(freq,T)
            else:
                vocabulary.__setitem__(word,(1,0))

    for sample in neg_train:
        for word in sample:
            word=word.lower().strip("'/\!@#$%&*+.:)(?-")
            if word == "":
                continue
            if vocabulary.__contains__(word):
                freq,T=vocabulary[word]
                freq=freq+1
                vocabulary[word]=(freq,T)
            else:
                vocabulary.__setitem__(word,(0,1))

    return vocabulary

def max_likelihood_estimation(pos_train,neg_train):
    pN=len(pos_train)
    nN=len(neg_train)
    N=pN+nN
    pCount=0
    nCount=0
    vocabulary=build_vocabulary(pos_train,neg_train)
    vN=len(vocabulary)

    for word in vocabulary:
        pCount=pCount+vocabulary[word][0]
        nCount=nCount+vocabulary[word][1]

    return {"voc":vocabulary,"pN":pN,"nN":nN,"N":N,"vN":vN,"pC":pCount,"nC":nCount}

def classifier(test_item,voc,pN,nN,N,vN,pC,nC):
    N=float(N)
    pr_pos=pN/N
    pr_neg=nN/N
    pos_pr=0.0
    neg_pr=0.0

    for token in test_item:
        token=token.lower().strip("'/\!@#$%&*+.:)(?-")
        if token=="":
            continue
        if voc.__contains__(token):
            pos_pr=pos_pr+math.log10(voc[token][0]+1.0/(pC+vN))
            neg_pr=neg_pr+math.log10(voc[token][1]+1.0/(nC+vN))
        else:
            pos_pr=pos_pr + math.log10(1.0/(pC+vN))
            neg_pr=neg_pr + math.log10(1.0/(nC+vN))

    pP=math.log10(pr_pos)+pos_pr
    nP=math.log10(pr_neg)+neg_pr

    return(pP,nP)

def testing(pos_test,neg_test,dic):
    conP=[[]]*2
    conN=[[]]*2
    conP[0]=0
    conP[1]=0
    conN[0]=0
    conN[1]=0
    for item in pos_test:
        pP,nP=classifier(item,dic['voc'],dic['pN'],dic['nN'],dic['N'],dic['vN'],dic['pC'],dic['nC'])
        if pP>nP:
            conP[0]+=1
        else:
            conP[1]+=1
        #print "PrbP",pP,nP

    for item in neg_test:
        pP,nP=classifier(item,dic['voc'],dic['pN'],dic['nN'],dic['N'],dic['vN'],dic['pC'],dic['nC'])
        if nP>pP:
            conN[0]+=1
        else:
            conN[1]+=1
        #print "PrbN",pP,nP

    print "confusion Matrix:"
    print "     neg pos"
    print "neg",conP
    print "pos",conN

    A=conP[1]+conN[0]
    B=conP[0]+conP[1]+conN[0]+conN[1]

    print "Error: ",(A*100.0)/B,"%"       
pos_dataset = read_review("pos","positve.txt")
neg_dataset = read_review("neg","negative.txt")
#print pos_dataset
pos_train,neg_train,pos_test,neg_test=split_dataset(pos_dataset,neg_dataset)
a=max_likelihood_estimation(pos_train,neg_train)
testing(pos_test,neg_test,a)
