import pandas as pd
import numpy as np
from funzioni import *
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# ------------------------ DEFINIZIONE VETTORE_ATTIVAZIONE E VETTORE_CAMION --------------------------------------
df_Coordinate = pd.read_csv('Node Matrix.csv')

# Vettore Attivazione: cassoni_campi
# Filtra il DataFrame per mantenere solo le righe con Label == 'c'
df_filtrato = df_Coordinate[df_Coordinate['Label'] == 'c']
# Seleziona solo le colonne 'ID' e 'Numero Cassoni'
Vettore_cassoni_campi_ID = df_filtrato[['ID', 'Number of Bins']].T

# Vettore Camion
df_filtrato_2 = df_Coordinate[df_Coordinate['Label'] == 'd']
# Seleziona solo le colonne 'ID' e 'Numero Camion'
Vettore_camion_ID = df_filtrato_2[['ID', 'Number of Trucks']].T

# Vettore Aziende ID
df_filtrato_3 = df_Coordinate[df_Coordinate['Label'] == 'a']
# Seleziona solo la colonne 'ID'
Vettore_aziende_ID = df_filtrato_3[['ID']].T

del df_filtrato, df_filtrato_2, df_filtrato_3

# ------------------------ ESTRAZIONE MATRICE DELLE DISTANZE (D) E VETTORE DELLE QUANTITA' (Q) -----------------------
# Leggi il file CSV in un DataFrame
df_distanze = pd.read_csv('Distance Matrix.csv', header=0, index_col=0)

# Sostituisci "inf" con np.inf
df_distanze.replace("inf", np.inf, inplace=True)

# Converti il DataFrame in un ndarray escludendo la prima riga e la prima colonna
D = df_distanze.to_numpy()
Q = Vettore_cassoni_campi_ID.iloc[1:].to_numpy()
C = Vettore_camion_ID.iloc[1:].to_numpy()

# ------------------------------ DEFINIZIONE PARAMETRI EURISTICI (ETA_D, ETA_C e TETA) -------------------------------
# Calcola il reciproco dei valori di D, tranne quelli che sono np.inf, quelli li pone uguali a zero
eta_d = np.where(D == np.inf, 0, 1 / D)
eta_c = 1 / Q

# ----------------------------- DEFINIZIONE CARATTERISTICHE FORMICA ---------------------------------------------------
capacita_camion = 88
N_Camion = int(np.ceil(np.sum(Q) / capacita_camion))
capacita_minima = int(np.ceil(np.sum(Q) / N_Camion))
MaxIt = 10  # Numero massimo di iterazioni
N_flotte = 10  # Numero di flotte da generare

# ------------------------------ DEFINIZIONE PARAMETRI METAEURISTICI (TAU) -------------------------------
# Calcola tau_0 (un numero) utilizzando la formula, e lo applica a tutta tau
D = np.where(D == np.inf, 0, D)  # Sostituisce inf con 0
tau0 = 10 * 1 / (D.shape[0] * np.matrix(D).mean())
# COSTRUZIONE MATRICE DEL FEROMONE
tau = tau0 * np.ones((D.shape[0], D.shape[1]))

tau_Q = tau0 * np.ones((capacita_camion, Q.shape[1]))

# Definizione dei parametri, alpha per il feromone, beta per l'euristica
alpha, beta = 1, 1
alpha_Q, beta_Q = 1, 1  # α > β => più numeri alti
rho = 0.05  # Tasso di evaporazione (Evaporation Rate)


class Ant:
    """Classe che rappresenta una singola formica (camion) con il proprio Tour e Cost."""

    def __init__(self, Tour: list[int] = None, Quantity: list[int] = None, Cost: float = 0):
        self.Tour = Tour if Tour is not None else []  # Lista delle città visitate
        self.Quantity = Quantity if Quantity is not None else []  # Lista delle quantità di ogni campo
        self.Cost = Cost  # Costo del percorso


class Ant_Fleet:
    """Classe che rappresenta una flotta di N_Camion formiche (camion)."""

    def __init__(self, num_ants: int = N_Camion, Cost: float = 0):
        self.ant = [Ant() for _ in range(num_ants)]  # Creiamo `num_ants` camion
        self.Cost = Cost  # Costo del percorso


Winner_fleet = Ant_Fleet(num_ants=0, Cost=99999)

# Ciclo che si ripete per 10 volte
for it in range(MaxIt):

    # ------------------------------- GENERAZIONE DELLE FLOTTE ------------------------------------------------------
    # Creazione di N_flotte di ant_fleet (fatte ognuna da num_ants) per ogni iterazione
    ant_fleet = [Ant_Fleet(num_ants=N_Camion, Cost=0) for _ in range(N_flotte)]

    # Ciclo per creare f flotte
    for f in range(N_flotte):

        ## PRIMA FLOTTA
        Q_fleet = Q.copy()
        C_fleet = C.copy()

        for k in range(N_Camion):
            # CALCOLO ETA DISTANZE
            # SELEZIONE RIGHE DI ETA DOVE CI SONO I CAMION: 1) Vedi quali sono le righe corrispondenti ai Camion, 2) a queste
            # rimuovi quelle dove i camion non ci sono, 3) A queste rimuovi quelle dove non ci sono cassoni nei campi, 4)
            # crea un eta di tutti 0 e mettici dentro solo le righe dove ci sono effettivamente i camion

            # 1) Seleziona la riga "ID" e convertila in array numerico
            id_values = Vettore_camion_ID.loc["ID"].to_numpy(dtype=int) - 1  # Sottrai 1 a ogni valore

            # Seleziona i valori della riga "Numero Camion"
            num_camion_values = Vettore_camion_ID.loc["Number of Trucks"].to_numpy(dtype=int)

            # Creazione di eta con tutte le celle a zero
            eta = np.zeros_like(eta_d)

            # Imposta i valori nelle colonne corrispondenti a id_values
            eta[id_values, :] = eta_d[id_values, :]  # Copia solo nelle colonne selezionate

            # ------------------------------------------------------------------------------------------------
            # ------------PRIMA SCELTA DELLE PRIMA FORMICA DELLA FLOTTA (SCELTA PRIMO CAMPO)------------------
            # RIMOZIONE
            # 2) Trova gli indici in cui "Numero Camion" e "Numero Cassoni" è 0
            zero_indices = id_values[num_camion_values == 0]  # Seleziona solo gli indici corrispondenti a camion = 0
            zero_indices_2 = np.where(Q_fleet == 0)[1]  # Seleziona solo gli indici corrispondenti a cassoni = 0
            # Imposta a zero anche le righe e le colonne corrispondenti in eta
            eta[zero_indices, :] = 0  # Rende nulle le righe corrispondenti
            eta[:, zero_indices_2] = 0  # Rende nulle le righe corrispondenti

            # Meta-Euristica --------------------------------------------------------------------
            P = (tau * alpha) * (
                        eta * beta)  # <----------------------------------------------- PARTE META-EURISTICA
            P = P / np.sum(P,
                           axis=None)  # <--------------------------------------------- PARTE META-EURISTICA
            riga_max, colonna_max = roulette_wheel_selection(P)  # <-- PARTE META-EURISTICA (Roulette selection)

            # Inserisci gli indici nella lista Tour della prima Ant di ant_fleet
            ant_fleet[f].ant[k].Tour.extend([int(riga_max), int(colonna_max)])

            # Inserisci il valore corrispondente di D nel Cost della prima Ant di ant_fleet
            ant_fleet[f].ant[k].Cost = D[riga_max, colonna_max]
            # -----------------------------------------------------------------------------------------------------
            # -------- PRIMA QUANTITA' DA RACCOGLIERE e UTILIZZO DEL PRIMO CAMION ---------------------------------
            ant_capacita_camion = capacita_camion
            # Trova l'indice nel DataFrame corrispondente a riga_max (che sarebbe il deposito di partenza)
            indice_df = np.where(Vettore_camion_ID.loc["ID"].to_numpy(dtype=int) - 1 == riga_max)[0]
            indice_df = indice_df[0]  # Prendi il primo valore trovato
            C_fleet[0, indice_df] -= 1  # Scala il valore nella colonna corrispondente

            # SCELTA QUANTITA' DA RACCOGLIERE SUL CAMPO CORRENTE
            # Prendi l'ultimo valore di Tour della prima Ant
            ultimo_nodo = ant_fleet[f].ant[k].Tour[-1]

            # Trova il valore corrispondente in Q_fleet
            valore_Q = Q_fleet[0, ultimo_nodo]
            if valore_Q == 0:
                a = 1

            quantita_euristica = int(min(valore_Q, ant_capacita_camion))  # <--------- TUTTA PARTE METAEURISTICA
            indice_df = np.where(Vettore_cassoni_campi_ID.loc["ID"].to_numpy(dtype=int) - 1 == ultimo_nodo)[0]
            indice_df = indice_df[0]  # Prendi il primo valore trovato
            tau_Q_colonna = tau_Q[:,
                            indice_df].copy()  # <------------------------------------ TUTTA PARTE METAEURISTICA
            # Imposta a zero gli elementi i cui indici sono > quantita_euristica
            tau_Q_colonna[np.arange(len(tau_Q_colonna)) > quantita_euristica - 1] = 0
            somma_tau_Q = np.sum(
                tau_Q_colonna)  # Calcola la sommatoria dei valori in tau_Q_colonna

            tau_Q_colonna *= alpha_Q  # <----------------------- TUTTA PARTE METAEURISTICA
            # Aggiungi la sommatoria nell'indice corrispondente a quantita_euristica
            tau_Q_colonna[quantita_euristica - 1] += somma_tau_Q * beta_Q
            # Controllo se tau_Q_colonna contiene NaN o Inf
            if (np.any(np.isnan(tau_Q_colonna)) or
                    np.any(np.isinf(tau_Q_colonna)) or
                    np.isnan(np.sum(tau_Q_colonna)) or
                    np.isinf(np.sum(tau_Q_colonna)) or
                    np.sum(tau_Q_colonna) == 0):
                a = 1
            P_Q = tau_Q_colonna / np.sum(
                tau_Q_colonna)  # Normalizza tau_Q_colonna per ottenere P_Q # <--------- TUTTA PARTE METAEURISTICA
            quantita_raccolta = int(roulette_wheel_selection_Q(P_Q))  # TUTTA PARTE METAEURISTICA (roulette selection)

            if quantita_raccolta > quantita_euristica:
                quantita_raccolta = quantita_euristica


            # Scala il valore di ant_capacita_camion e di Q_fleet
            ant_capacita_camion -= quantita_raccolta
            Q_fleet[0, ultimo_nodo] -= quantita_raccolta  # Riduci anche il valore in Q_fleet
            # Aggiungi il valore raccolto dalla formica in quel campo a Quantità
            ant_fleet[f].ant[k].Quantity.append(quantita_raccolta)

            # La scrittura di questo while in questo modo indica che il ciclo continua a girare se entrambe le condizioni sono vere
            while not (0 <= ant_capacita_camion <= (capacita_camion - capacita_minima)) and np.sum(
                    Q_fleet) != 0:  # capacita_camion - capacita_minima = spazion per 7 cassoni vuoto
                # CAMBIO CAMPO: RI-CREAZIONE DI ETA
                # Estrai la riga da eta_d corrispondente all'indice di ultimo_nodo
                eta_riga = eta_d[ultimo_nodo, :Q_fleet.shape[1]].flatten()  # Estrai fino alla lunghezza di Q_fleet
                tau_riga = tau[ultimo_nodo, :Q_fleet.shape[1]].flatten()

                # Porta a zero le celle in eta_riga dove Q_fleet ha valore zero
                eta_riga[Q_fleet[0, :] == 0] = 0  # Modifica eta_riga portando a zero le posizioni corrispondenti

                P_riga = (tau_riga * alpha) * (eta_riga * beta)  # <--------------------------- PER METAEURISTICA
                if np.all(P_riga == 0):
                    indice_max = ant_fleet[f].ant[k].Tour[-1]
                else:
                    P_riga = P_riga / np.sum(P_riga, axis=None)  # <--------------------------- PER METAEURISTICA
                    indice_max = int(roulette_wheel_selection_Q(P_riga))-1   # PER METAEURISTICA (roulette selection)

                # Aggiungi l'indice di eta_riga al Tour della prima formica (ant_fleet.ant[0])
                ant_fleet[f].ant[k].Tour.append(indice_max)

                # Inserisci il valore corrispondente di D nel Cost della prima Ant di ant_fleet
                ant_fleet[f].ant[k].Cost += D[
                    ultimo_nodo, indice_max]  # Aggiorna il costo con il valore di D per l'indice minimo

                # SCELTA QUANTITA' DA RACCOGLIERE SUL CAMPO CORRENTE
                # Prendi l'ultimo valore di Tour della prima Ant
                ultimo_nodo = ant_fleet[f].ant[k].Tour[-1]
                # Trova il valore corrispondente in Q_fleet
                valore_Q = Q_fleet[0, ultimo_nodo]

                # Scegli un valore intero casuale tra 1 e il minimo tra valore_Q e ant_capacita_camion  <----------------INIZIO PARTE METAEURISTICA
                # Generazione del numero con l'influenza delle formiche
                quantita_euristica = int(min(valore_Q, ant_capacita_camion))  # <------------ PARTE METAEURISTICA
                indice_df = np.where(Vettore_cassoni_campi_ID.loc["ID"].to_numpy(dtype=int) - 1 == ultimo_nodo)[0]
                indice_df = indice_df[0]  # Prendi il primo valore trovato    #  <-------------- PARTE METAEURISTICA
                tau_Q_colonna = tau_Q[:, indice_df].copy()  # <------------------------------ PARTE METAEURISTICA
                # Imposta a zero gli elementi i cui indici sono > quantita_euristica
                tau_Q_colonna[np.arange(len(tau_Q_colonna)) > quantita_euristica - 1] = 0
                # Calcola la sommatoria dei valori in tau_Q_colonna   <-- PARTE METAEURISTICA
                somma_tau_Q = np.sum(tau_Q_colonna)
                # Aggiungi la sommatoria nell'indice corrispondente a quantita_euristica
                tau_Q_colonna *= alpha_Q  # <------------------------------ PARTE METAEURISTICA
                tau_Q_colonna[quantita_euristica - 1] += somma_tau_Q * beta_Q
                P_Q = tau_Q_colonna / np.sum(tau_Q_colonna)  # Normalizza tau_Q_colonna per ottenere P_Q
                quantita_raccolta = int(roulette_wheel_selection_Q(P_Q))    # <------------------------------ PARTE METAEURISTICA (roulette selection)


                # Scala il valore di ant_capacita_camion e di Q_fleet
                ant_capacita_camion -= quantita_raccolta
                Q_fleet[0, ultimo_nodo] -= quantita_raccolta  # Riduci anche il valore in Q_fleet
                # Aggiungi il valore raccolto dalla formica in quel campo a Quantità
                ant_fleet[f].ant[k].Quantity.append(quantita_raccolta)

            # Trova l'ultimo valore della lista Tour della prima formica
            ultimo_valore_tour = ant_fleet[f].ant[k].Tour[-1]

            # Seleziona gli indici delle colonne corrispondenti ai valori numerici presenti nel DataFrame Vettore_aziende_ID
            colonne_eta_d = Vettore_aziende_ID.loc["ID"].to_numpy(
                dtype=int) - 1  # Sottrai 1 per indicizzazione zero-based

            # Seleziona il sottoarray di eta_d: riga corrispondente a ultimo_valore_tour e colonne selezionate
            sottoarray_eta_d = eta_d[ultimo_valore_tour, colonne_eta_d]

            # Trova il valore minimo nel sottoarray e il suo indice relativo
            valore_minimo = np.max(sottoarray_eta_d)  # Trova il valore numerico minimo
            indice_relativo = np.argmax(sottoarray_eta_d)  # Indice relativo nel sottoarray

            # Calcola l'indice corretto nella matrice eta_d sommando il minimo valore di Vettore_aziende_ID
            indice_eta_d = int(indice_relativo + (np.min(Vettore_aziende_ID.loc["ID"].to_numpy(dtype=int)) - 1))

            # Aggiungi l'indice di eta_riga al Tour della prima formica (ant_fleet.ant[0])
            ant_fleet[f].ant[k].Tour.append(indice_eta_d)

            # Inserisci il valore corrispondente di D nel Cost della prima Ant di ant_fleet
            ant_fleet[f].ant[k].Cost += D[
                ultimo_valore_tour, indice_eta_d]  # Aggiorna il costo con il valore di D per l'indice minimo

        # Calcola la somma dei costi delle formiche nella flotta
        ant_fleet[f].Cost = sum(ant.Cost for ant in ant_fleet[f].ant)

        # Formica 1 della flotta 1 di iterazione 1 è la prima vincitrice
        if it == 0 and f == 0:
            Winner_fleet = ant_fleet[0]

        # Aggiornamento flotta formiche vincitrice
        if ant_fleet[f].Cost < Winner_fleet.Cost:
            Winner_fleet = ant_fleet[f]

    # AGGIORNAMENTO FEROMONE DISTANZE utilizzando TUTTE le formiche
    for f in range(N_flotte):
        for k in range(N_Camion):
            for t in range(0, len(ant_fleet[f].ant[k].Tour) - 1, 1):
                i = ant_fleet[f].ant[k].Tour[t]
                j = ant_fleet[f].ant[k].Tour[t + 1]

                tau[i][j] = tau[i][j] + 1 / ant_fleet[f].Cost

    # AGGIORNAMENTO FEROMONE QUANTITA' utilizzando TUTTE le formiche
    for f in range(N_flotte):
        for k in range(N_Camion):
            Tour_Raccolta = ant_fleet[f].ant[k].Tour[1:-1]
            Quantita_Raccolta = ant_fleet[f].ant[k].Quantity

            # Creiamo un dizionario per sommare le quantità per ogni valore unico in Tour_Raccolta
            from collections import defaultdict

            somma_quantita = defaultdict(int)
            for tour, quantita in zip(Tour_Raccolta, Quantita_Raccolta):
                somma_quantita[tour] += quantita

            # Convertiamo il dizionario in liste
            Tour_unici = list(somma_quantita.keys())  # [4, 5]
            Somme_quantita = list(somma_quantita.values())  # [10, 27]

            # Aggiungiamo 0.05 nelle posizioni corrispondenti
            for quantita, tour in zip(Somme_quantita, Tour_unici):
                tau_Q[quantita-1, tour] += 1 / ant_fleet[f].Cost  # Riga: valore della somma, Colonna: valore unico di Tour_Raccolta


    # EVAPORAZIONE FEROMONE
    tau = (1 - rho) * tau
    tau_Q = (1 - rho) * tau_Q


# Salvataggio dell'oggetto ant_fleet in un file binario
with open('Winner_fleet.pkl', 'wb') as f:
    pickle.dump(Winner_fleet, f)

print("\n===== RISULTATO FINALE =====")
print(f"Costo della flotta vincitrice: {Winner_fleet.Cost}")