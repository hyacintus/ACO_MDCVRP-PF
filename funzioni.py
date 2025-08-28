import random
import numpy as np
from datetime import datetime


# Funzione dove viene definita quale coppia di nodi dispari prendere
def roulette_wheel_selection(p):
    # Pezzo do codice che rinnova ogni volta i numeri casuali in modo che ci sia reale casualità dei numeri
    random.seed()
    np.random.seed(int(datetime.now().timestamp()))
    r = random.uniform(0, 1)
    while r == 0:
        r = random.uniform(0, 1)  # Non voglio che mi capiti la scelta Nodo Partenza = Arrivo = 0

    p_vector = p.flatten(order='C')
    c = np.cumsum(p_vector)

    # get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
    # Numeri = get_indexes(r, C)
    indici_minori = np.where(c <= r)
    j = max(indici_minori[0]) + 1
    # Quando ci si muove con le matrici si devono indicare tutti e 2 gli indici, se no ti da errore
    # j = C[0, Massimo+1]
    # Restituisce riga e colonna da rimuovere della matrice P
    i = int(j / len(p))
    j = j % len(p)
    return i, j


def roulette_wheel_selection_Q(p):
    # Pezzo do codice che rinnova ogni volta i numeri casuali in modo che ci sia reale casualità dei numeri
    random.seed()
    np.random.seed(int(datetime.now().timestamp()))
    c = np.cumsum(p)
    r = random.uniform(0, float(c[-1]))
    while r == 0:
        r = random.uniform(0, float(c[-1]))  # Non voglio che mi capiti la scelta Nodo Partenza = Arrivo = 0

    indici_minori = np.where(c <= r)
    # Verifica se gli indici trovati sono vuoti, in tal caso imposta j a 0

    if len(indici_minori[0]) == 0: # Controlla se l'array è vuoto
        j = 0
    else:
        j = max(indici_minori[0]) + 1  # Ottieni l'indice massimo tra quelli minori o uguali a r

    return j + 1


def elite_selection(p):
    massimo = np.max(p)
    indici_massimi = np.where(p == massimo)[0]  # Trova tutti gli indici del massimo
    destinazione = int(indici_massimi[-1])  # Prendi l'ultimo

    return destinazione


def elite_selection_Q(p):
    massimo = np.max(p)
    indici_massimi = np.where(p == massimo)[0]  # Trova tutti gli indici del massimo
    q_raccolta = int(indici_massimi[-1] + 1)  # Prendi l'ultimo

    return q_raccolta
