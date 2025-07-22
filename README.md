# Progetto di Analisi Statistica: Dataset Penguins

## 📋 Descrizione del Progetto

Questo progetto analizza il dataset `penguins.csv` utilizzando tecniche di statistica descrittiva e inferenziale. L'obiettivo è esplorare le relazioni tra le caratteristiche fisiche di diverse specie di pinguini e costruire modelli di machine learning per predire la specie di un pinguino in base alle sue misure.

Il dataset raccoglie informazioni su tre specie di pinguini (Adelie, Chinstrap, Gentoo) da tre isole dell'arcipelago Palmer (Biscoe, Dream, Torgersen).

## 🛠️ Struttura dell'Analisi

Il progetto segue una pipeline di analisi dati strutturata in più fasi:

1.  **Caricamento e Pre-Processing**: Il dataset viene caricato e pulito rimuovendo i valori mancanti per garantire la qualità dei dati.
2.  **Exploratory Data Analysis (EDA)**: Vengono utilizzati istogrammi, matrici di correlazione e grafici bivariati per esplorare le distribuzioni delle variabili e le relazioni tra di esse.
3.  **Regressione Lineare**: Viene costruito un modello di regressione per studiare la relazione tra la lunghezza delle pinne (`flipper_length_mm`) e la massa corporea (`body_mass_g`), due variabili che mostrano una forte correlazione positiva (0.87).
4.  **Classificazione e Ottimizzazione**: Vengono addestrati modelli di classificazione (Regressione Logistica e SVM) per predire la specie del pinguino. Gli iperparametri del modello SVM vengono ottimizzati per trovare la configurazione più performante.
5.  **Valutazione Statistica**: Le performance del modello finale vengono validate in modo robusto ripetendo l'esperimento più volte per valutarne la stabilità e l'affidabilità.

## 🚀 Come Eseguire il Codice

Per eseguire l'analisi sul tuo computer, segui questi passaggi.

### Prerequisiti

Assicurati di avere Python installato. Poi, installa le librerie necessarie tramite pip. Puoi creare un file `requirements.txt` con il seguente contenuto e poi eseguire `pip install -r requirements.txt` dal tuo terminale.

**File `requirements.txt`:   
pandas
seaborn
matplotlib
scikit-learn
statsmodels
numpy
**

### Esecuzione

Una volta installate le librerie, esegui semplicemente lo script Python dalla riga di comando:
```bash
python penguins.py
