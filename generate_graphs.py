
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



# segmentare binara
#INPUT_PATH = r'E:\an_4_LICENTA\Scripturi_24mar\Scripturi\core\results\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_AugumentareFLIP,ZOOM_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bil0.9369.txt'
# dreapta (jos)
INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\results\wt1.txt' # cu augumentare
# stanga (jos)
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_binar_brats_2D_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bilinear_0.9316.txt' # 
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_RESNET_HYBRID_1_number_of_blocks_3.txt' # OVERFIT SAU ZGOMOT?

# segmentare multiclasa
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\results_semantic\ABSOLUTE_FULL_DTS__unet_semantic_brats_2D_v3_COPIE_sigmaoida_fcost_dice_Vlad_weights_batch_gibbsremove_eph61.txt'
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\results_semantic\delazero.txt'
#INPUT_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\results_semantic\transferLR_gibbsoptimization_1.txt'


INPUT_PATH1 = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\results_semantic\delazero.txt'
INPUT_PATH2 = r'E:\an_4_LICENTA\Workspace\Scripturi\core_semantic\results_semantic\UNET_binar_brats_2D_FULLDATASET_lr=10^(-4)CONSTANT_Augumentare_FLIP,ZOOM_GAUSS_de_la_inceput_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bil.txt'

INPUT_PATH1 = r'E:\an_4_LICENTA\Workspace\Scripturi\core\results\wt1.txt' 
INPUT_PATH2 = r'E:\an_4_LICENTA\Workspace\Scripturi\core\results\UNET_binar_brats_2D_Gibbsringingremoved_Squeeze&Extract_biassfieldremoved2D_8b_SIGMOIDA_32_64_128_256_lr=10^(-4)_normalizaresimpla_bilinear_0.9316.txt' # 


def extract_data(INPUT_PATH):

    A = [] # lista cu atribute, adica cu metricile
    # = [] # de fapt nu prea am nevoie de x
    Y = [] # va fi omlista de liste. fiecare lista contine scorurile, de-a lungul axei timpului

    with open(INPUT_PATH, 'r') as file:
        text = file.read()

    lines = text.splitlines()

    for line in lines:
        line_sep_by_coma = line.split(',')
        for element in line_sep_by_coma:
            if len(element.split(':')) == 2:
                attribute, value = element.split(':')
                attribute, value = attribute.strip(), value.strip()
                # se trateaza cazul in care nu se regaseste inca acest atrinut inlista cu atribute 
                if attribute not in A:
                    A.append(attribute)
                if len(Y) <= A.index(attribute):
                    Y.append([])
                Y[A.index(attribute)].append(float(value))
    return A, Y

#print(Y)
#print(len(Y))

def generate_2_overlaped_graphs(INPUT_PATH1, INPUT_PATH2):
    A1, Y1 = extract_data(INPUT_PATH1)
    A2, Y2 = extract_data(INPUT_PATH2)
    for y1, y2 in zip(Y1, Y2):
        plt.figure()
        print(y1)
        plt.plot([i for i in range(1, len(y1)+1)], list(y1), label='cu augmentare')
        plt.plot([i for i in range(1, len(y2)+1)], list(y2), label='fără augmentare')
        plt.xlabel('Număr de epoci')
        #plt.ylabel('Valoarea funcției de cost')
        plt.ylabel('Dice - Validare')
        plt.legend(loc='upper right', fontsize=12)
        #plt.legend(loc='lower right', fontsize=12)
        plt.show()

def generate_one_graph(INPUT_PATH):
    A, Y = extract_data(INPUT_PATH)
    for y in Y:
        plt.figure()
        print(y)
        plt.plot([i for i in range(1, len(y)+1)], list(y))
        plt.show()

def generate_graph_binara_val_vs_train_dice(INPUT_PATH):
    A, Y = extract_data(INPUT_PATH)
    plt.figure()
    train_dice = []
    for i in Y[0]:
        train_dice.append(1-i)
    plt.plot([i for i in range(1, len(train_dice)+1)], list(train_dice), label='F1-antrenare')
    plt.plot([i for i in range(1, len(Y[2])+1)], list(Y[2]), label='F1-validare')
    plt.xlabel('Număr de epoci')
    plt.ylabel('Dice')
    plt.legend(loc='lower left', fontsize=12)
    plt.show()


#################################

# Geenrate one graph
#generate_one_graph(INPUT_PATH)

# Generate 2 graphs
generate_2_overlaped_graphs(INPUT_PATH1, INPUT_PATH2)

# Genereaza grafic pentru train dice vs val dice - ca sa se vada overfitul
#generate_graph_binara_val_vs_train_dice(INPUT_PATH)









