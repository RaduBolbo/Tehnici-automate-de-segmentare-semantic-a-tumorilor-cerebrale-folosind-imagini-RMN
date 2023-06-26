import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

#DIR_1 = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\PNG_Training_Dataset'
#DIR_2 = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\biassfieldremoved_PNG_Train_Dataset'

DIR_1 = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_nepreprocesat\PNG_Training_Dataset'
#DIR_2 = r'E:\an_4_LICENTA\Workspace\Dataset\Full_Dataset\gibremoved_PNG_Train_Dataset'
DIR_2 = r'E:\an_4_LICENTA\Workspace\Dataset\Optimization_Dataset_biassfieldremoved\biassfieldremoved_2D_PNG_Train_Dataset'

no_example = '00008'
#type = 't1' # pare foarte bine
type = 'flair'

# alegerea numarului de moduri
#nr_moduri = 2
nr_moduri = 4

PI = 3.1415
eps = 0.00000001
L=254

# functie care genereaza valorile unei/unor distributii gaussiene pentru care se dau medie, varianta, probabilitate apriori
def construct_gaussian_mixture(m,v,prob): # m, v, prob sunt VECTORI !!!!
    K=np.size(m)
    prob=prob/np.sum(prob)
    pdf=np.zeros([254,K]) # pdf va contine distributiile individuale
    coef=prob/(np.sqrt(2*PI*v)) # ce e in fata gaussienei
    for i in range(0,254):
        pdf[i,:]=coef*np.exp(-(i-m)*(i-m)/(2*v)) # se pune la vlaoarea i din pdf 
    hmixt = np.sum(pdf,axis=1) # se sumeaza distributiile stocate in pdf
    hmixt=hmixt/sum(hmixt) # se asigura ca este distributie de probabilitate
    return hmixt, pdf

# functia care face descompunerea unei histograme in moduri gaussiene
def ExpMax(hist,K): # K = nr de gaussiene care se cauta
    p=np.ones(K)/K
    mu=L*np.array(range(1,K+1))/(K+1)
    v= L*np.ones(K)
    while(True):
        hmixt, prb = construct_gaussian_mixture(mu,v,p)
        scal = np.sum(prb,axis=1)+eps #prob sunt distributiile individuLE ESTIMATE
        loglik=np.sum(hist*np.log(scal))
        for i in range(0,K):
            pp=hist*prb[:,i]/scal
            p[i]=np.sum(pp)
            mu[i]=np.sum(np.array(range(0,L))*pp)/p[i]
            vr=np.array(range(0,L))-mu[i]
            v[i]=np.sum(vr*vr*pp)/p[i]
        p=p/np.sum(p)

        hmixt,prb = construct_gaussian_mixture(mu,v,p)
        scal = np.sum(prb,axis=1)+eps
        nloglik=np.sum(hist*np.log(scal))
        if((nloglik-loglik)<0.0001):
            break
    #gata while
    return mu, v, p

def check_nakagawa(h, mu0, mu1, v0, v1): # CRED ca pentru Nakagama am nevoie deja de un prag
        # SE PRESUPUNE CA nivelurile de gri SUNT IN np.uint8 !!!!!
        
        # acum abia verific conditia a 3-a:
        termen_dreapta = 0.8*min(h[np.uint8(mu0)], h[np.uint8(mu1)])
        #print(termen_dreapta)
        termen_stanga = np.min(h[np.uint8(mu0):np.uint8(mu1)+1])
        #print(termen_stanga)
        
        print('distanta intre medii: ')
        print(abs(mu0-mu1))
        print('raportul variantelor: ')
        print(math.sqrt(v1/v0))
        print('minuimul dintre moduri:')
        print(min(h[np.uint8(mu0)], h[np.uint8(mu1)]))

        if abs(mu0-mu1) > 4 and math.sqrt(v1/v0) < 10 and math.sqrt(v1/v0) > 0.1 and termen_dreapta > termen_stanga:
            return True
        else:
            return False

def compute_histogram_of_example_x(DIR, no_example, type):
    arr = np.zeros([240, 240, 150])
    i = 0
    for dir_name in os.listdir(DIR):
        if no_example in dir_name:
            filepath = DIR + '\\' + dir_name + '\\' + type + '.png'
            img = cv2.imread(filepath)
            arr[:, :, i] = img[:, :, 0]
            i += 1

    hist = np.histogram(arr, bins=256, range=[0, 256], density = False)
    hist = hist[0][1:-1]/np.sum(hist[0][1:-1])
    return hist
        
def compute_histogram_of_the_entire_dataset(DIR, no_example, type):
    arr = np.zeros([240, 240, 10000])
    i = 0
    ex=0
    #nr_max = 100
    nr_max = 400
    for dir_name in os.listdir(DIR):
        if ex < nr_max:
            filepath = DIR + '\\' + dir_name + '\\' + type + '.png'
            img = cv2.imread(filepath)
            arr[:, :, i] = img[:, :, 0]
            i += 1
        ex += 1

    hist = np.histogram(arr, bins=256, range=[0, 256], density = False)
    hist = hist[0][1:-1]/np.sum(hist[0][1:-1])
    return hist

hist1 = compute_histogram_of_example_x(DIR_1, no_example, type)

m1, v1, p1 = ExpMax(hist1, nr_moduri)
hst1, pr1 = construct_gaussian_mixture(m1, v1, p1)

plt.figure()
plt.plot(hist1), plt.plot(pr1), plt.show()

hist2 = compute_histogram_of_example_x(DIR_2, no_example, type)

m2, v2, p2 = ExpMax(hist2, nr_moduri)
hst2, pr2 = construct_gaussian_mixture(m2, v2, p2)

plt.figure()
plt.plot(hist2), plt.plot(pr2), plt.show()


####################################################### PE unj procent din dataset


hist1 = compute_histogram_of_the_entire_dataset(DIR_1, no_example, type)

plt.figure()
plt.plot(hist1), plt.show()

hist2 = compute_histogram_of_the_entire_dataset(DIR_2, no_example, type)

plt.figure()
plt.plot(hist2), plt.show()

#### acum se suprapun peste histograme estimatele celor 3 gaussiene

m1, v1, p1 = ExpMax(hist1, nr_moduri)
hst1, pr1 = construct_gaussian_mixture(m1, v1, p1)

plt.figure()
plt.plot(hist1), plt.plot(pr1), plt.show()

m2, v2, p2 = ExpMax(hist2, nr_moduri)
hst2, pr2 = construct_gaussian_mixture(m2, v2, p2)

plt.figure()
plt.plot(hist2), plt.plot(pr2), plt.show()

print(m1, v1)
print(m2, v2)

print(check_nakagawa(hist1, m1[1], m1[2], v1[1], v1[2]))

print(check_nakagawa(hist2, m2[1], m2[2], v2[1], v2[2]))






