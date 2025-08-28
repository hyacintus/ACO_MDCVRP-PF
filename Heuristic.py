import pandas as pd
import numpy as np
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
# Definizione dei parametri
alpha_skew, beta_skew = 2, 0.5  # α > β => più numeri alti

# ----------------------------- DEFINIZIONE CARATTERISTICHE FORMICA ---------------------------------------------------
capacita_camion = 88
N_Camion = int(np.ceil(np.sum(Q) / capacita_camion))
capacita_minima = int(np.ceil(np.sum(Q) / N_Camion))
MaxIt = 1  # Numero massimo di iterazioni
N_flotte = 1  # Numero di flotte da generare


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

            # RIMOZIONE
            # 2) Trova gli indici in cui "Numero Camion" e "Numero Cassoni" è 0
            zero_indices = id_values[num_camion_values == 0]  # Seleziona solo gli indici corrispondenti a camion = 0
            zero_indices_2 = np.where(Q_fleet == 0)[1]  # Seleziona solo gli indici corrispondenti a cassoni = 0
            # Imposta a zero anche le righe e le colonne corrispondenti in eta
            eta[zero_indices, :] = 0  # Rende nulle le righe corrispondenti
            eta[:, zero_indices_2] = 0  # Rende nulle le righe corrispondenti
            eta = eta / np.sum(eta)

            # PRIMA SCELTA DELLE PRIMA FORMICA DELLA FLOTTA (SCELTA PRIMO CAMPO)
            # Trova gli indici della cella con il valore massimo in eta
            indice_max = np.unravel_index(np.argmax(eta), eta.shape)
            riga_max, colonna_max = indice_max

            # Inserisci gli indici nella lista Tour della prima Ant di ant_fleet
            ant_fleet[f].ant[k].Tour.extend([int(riga_max), int(colonna_max)])

            # Inserisci il valore corrispondente di D nel Cost della prima Ant di ant_fleet
            ant_fleet[f].ant[k].Cost = D[riga_max, colonna_max]

            # PRIMA QUANTITA' DA RACCOGLIERE e UTILIZZO DEL PRIMO CAMION
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

            # Generazione del numero casuale scalato tra 1 e max_val
            quantita_raccolta = int(min(valore_Q, ant_capacita_camion))

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

                # Porta a zero le celle in eta_riga dove Q_fleet ha valore zero
                eta_riga[Q_fleet[0, :] == 0] = 0  # Modifica eta_riga portando a zero le posizioni corrispondenti
                # Porta a zero i valori in eta_riga corrispondenti agli indici in ant_fleet.ant[0].Tour
                tour_indices = [idx for idx in ant_fleet[f].ant[k].Tour if
                                idx <= Q_fleet.shape[1] - 1]  # Prendi solo numeri validi
                eta_riga[tour_indices] = 0  # Imposta a zero gli indici corrispondenti

                eta_riga /= np.sum(eta_riga)

                # Trova il valore minimo di eta_riga e l'indice corrispondente
                indice_max = int(np.argmax(eta_riga))

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
                # Scegli un valore intero casuale tra 1 e il minimo tra valore_Q e ant_capacita_camion
                quantita_raccolta = int(min(valore_Q, ant_capacita_camion))
                # Scala il valore di ant_capacita_camion e di Q_fleet
                ant_capacita_camion -= quantita_raccolta
                Q_fleet[0, ultimo_nodo] -= quantita_raccolta  # Riduci anche il valore in Q_fleet
                # Aggiungi il valore raccolto dalla formica in quel campo a Quantità
                ant_fleet[f].ant[k].Quantity.append(quantita_raccolta)

            # Trova l'ultimo valore della lista Tour della prima formica
            ultimo_valore_tour = ant_fleet[f].ant[k].Tour[-1]

            # Seleziona gli indici delle colonne corrispondenti ai valori numerici presenti nel DataFrame Vettore_aziende_ID
            colonne_eta_d = Vettore_aziende_ID.loc["ID"].to_numpy(dtype=int) - 1  # Sottrai 1 per indicizzazione zero-based

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


# Salvataggio dell'oggetto ant_fleet in un file binario
with open('ant_fleet.pkl', 'wb') as f:
    pickle.dump(ant_fleet, f)