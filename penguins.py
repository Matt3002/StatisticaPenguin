import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import statsmodels.api as sm

print("Esecuzione del progetto di analisi sul dataset Penguins.\n")

# --- PUNTO 1 & 2: Caricamento e Pre-Processing ---
print("--- FASE 1 & 2: Caricamento e Pre-Processing del Dataset ---")

# Seaborn ha un comodo metodo per caricare il dataset penguins
# In alternativa, si potrebbe usare: df = pd.read_csv("penguins.csv")
df = sns.load_dataset("penguins")

# Rimuoviamo le righe con valori mancanti, come descritto nel PDF
df.dropna(inplace=True)

print("Dataset caricato e pulito. Numero di righe rimaste:", len(df))
print("Prime 5 righe del dataset pulito:")
print(df.head())
print("-" * 50, "\n")


# --- PUNTO 3: Exploratory Data Analysis (EDA) ---
print("--- FASE 3: Analisi Esplorativa dei Dati (EDA) ---")

# 1. Istogrammi delle variabili numeriche
print("Visualizzazione degli istogrammi delle variabili numeriche...")
df.hist(figsize=(12, 10), bins=20)
plt.suptitle("Distribuzione delle Variabili Numeriche", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 2. Matrice di correlazione
print("Visualizzazione della matrice di correlazione...")
numeric_cols = df.select_dtypes(include=np.number)
corr_matrix = numeric_cols.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Matrice di Correlazione tra Variabili Numeriche")
plt.show()

# 3. Grafico bivariato (Pair Plot) colorato per specie
print("Visualizzazione del grafico bivariato (Pair Plot)...")
sns.pairplot(df, hue='species', palette='colorblind')
plt.suptitle("Analisi Bivariata delle Variabili per Specie", y=1.02)
plt.show()
print("-" * 50, "\n")


# --- PUNTO 4: Splitting del Dataset ---
print("--- FASE 4: Divisione del Dataset in Training, Validation e Test ---")

# Variabile target (da predire)
target_var = 'species'
X = df.drop(columns=target_var)
y = df[target_var]

# Gestiamo le variabili categoriche (non numeriche) con One-Hot Encoding
# Dobbiamo farlo prima di dividere per assicurarci che tutte le colonne siano presenti
X_encoded = pd.get_dummies(X, drop_first=True)

# Divisione: 60% training, 40% temporaneo (che poi divideremo a metà)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_encoded, y, test_size=0.4, random_state=42, stratify=y
)

# Divisione del set temporaneo: 20% validation, 20% test
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Dimensioni del Training Set: {X_train.shape}")
print(f"Dimensioni del Validation Set: {X_val.shape}")
print(f"Dimensioni del Test Set: {X_test.shape}")
print("-" * 50, "\n")


# --- PUNTO 5: Regressione Lineare ---
print("--- FASE 5: Regressione Lineare ---")
print("Eseguo la regressione tra 'flipper_length_mm' e 'body_mass_g'.")

# Prepariamo i dati per la regressione dal DataFrame originale
X_reg = df[['flipper_length_mm']]
y_reg = df['body_mass_g']

# Aggiungiamo la costante per il calcolo dell'intercetta con statsmodels
X_reg_sm = sm.add_constant(X_reg)

# Creiamo e addestriamo il modello OLS (Ordinary Least Squares)
model_sm = sm.OLS(y_reg, X_reg_sm).fit()

# Stampa del riassunto del modello (contiene R^2, coefficienti, etc.)
print("\nRiassunto del Modello di Regressione Lineare:")
print(model_sm.summary())

# Calcolo dell'MSE
y_pred_reg = model_sm.predict(X_reg_sm)
mse = mean_squared_error(y_reg, y_pred_reg)
print(f"\nErrore Quadratico Medio (MSE): {mse}")

# Grafico della retta di regressione
plt.figure(figsize=(10, 6))
plt.scatter(X_reg, y_reg, label='Dati reali')
plt.plot(X_reg, y_pred_reg, color='red', linewidth=2, label='Retta di regressione')
plt.title('Regressione Lineare: Lunghezza Pinne vs. Massa Corporea')
plt.xlabel('Flipper Length (mm)')
plt.ylabel('Body Mass (g)')
plt.legend()
plt.grid(True)
plt.show()

# Analisi dei residui
residuals = model_sm.resid
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20)
plt.title('Distribuzione dei Residui del Modello')
plt.xlabel('Residuo')
plt.ylabel('Frequenza')
plt.grid(True)
plt.show()
print("-" * 50, "\n")


# --- PUNTO 6: Addestramento Modelli di Classificazione ---
print("--- FASE 6: Addestramento Modelli di Classificazione ---")

# 1. Regressione Logistica
print("\nAddestramento Regressione Logistica...")
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_val_pred_lr = log_reg.predict(X_val)
acc_lr = accuracy_score(y_val, y_val_pred_lr)
print(f"Accuratezza Regressione Logistica su Validation Set: {acc_lr:.2f}")

# 2. SVM con vari kernel
print("\nAddestramento SVM con vari kernel...")
kernels = ['linear', 'poly', 'rbf']
for kernel in kernels:
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train)
    y_val_pred_svm = svm_model.predict(X_val)
    acc_svm = accuracy_score(y_val, y_val_pred_svm)
    print(f"Accuratezza SVM con kernel '{kernel}' su Validation Set: {acc_svm:.2f}")
print("-" * 50, "\n")


# --- PUNTO 7: Hyperparameter Tuning ---
print("--- FASE 7: Hyperparameter Tuning per SVM ---")

# Definiamo la griglia dei parametri da provare
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1]
}

# Usiamo GridSearchCV per trovare i migliori iperparametri
# GridSearchCV usa la cross-validation, ma per coerenza con il PDF, lo addestriamo su train e valutiamo su val
print("Ricerca dei migliori iperparametri con GridSearchCV...")
grid_search = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=0, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Risultati del tuning
print(f"Migliori parametri trovati: {grid_search.best_params_}")
best_accuracy = grid_search.best_score_
print(f"Migliore accuratezza durante la cross-validation: {best_accuracy:.2f}")

# Il miglior modello è già addestrato
best_svm_model = grid_search.best_estimator_
print("-" * 50, "\n")


# --- PUNTO 8: Valutazione della Performance sul Test Set ---
print("--- FASE 8: Valutazione del Modello Finale sul Test Set ---")

y_test_pred = best_svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Accuratezza del miglior modello SVM sul Test Set: {test_accuracy:.4f}")

# Classification Report
print("\nClassification Report sul Test Set:")
print(classification_report(y_test, y_test_pred))

# Matrice di Confusione
print("\nMatrice di Confusione sul Test Set:")
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=best_svm_model.classes_, yticklabels=best_svm_model.classes_)
plt.title('Matrice di Confusione sul Test Set')
plt.xlabel('Etichetta Prevista')
plt.ylabel('Etichetta Reale')
plt.show()
print("-" * 50, "\n")


# --- PUNTO 9: Studio Statistico sui Risultati ---
print("--- FASE 9: Studio Statistico della Stabilità del Modello ---")

k = 10
accuracies = []

print(f"Ripetizione dell'addestramento e test per {k} volte...")

for i in range(k):
    # Dividiamo i dati ogni volta con un random_state diverso
    X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(X_encoded, y, test_size=0.2, random_state=i, stratify=y)
    
    # Usiamo il modello con i migliori iperparametri trovati prima
    model_k = SVC(**grid_search.best_params_, random_state=i)
    model_k.fit(X_train_k, y_train_k)
    
    y_pred_k = model_k.predict(X_test_k)
    accuracies.append(accuracy_score(y_test_k, y_pred_k))

# Calcolo di media e deviazione standard
mean_accuracy = np.mean(accuracies)
std_accuracy = np.std(accuracies)

print(f"\nAccuratezza Media su {k} iterazioni: {mean_accuracy:.4f}")
print(f"Deviazione Standard dell'Accuratezza su {k} iterazioni: {std_accuracy:.4f}")

# Grafici della distribuzione delle accuratezze
plt.figure(figsize=(14, 6))

# Istogramma
plt.subplot(1, 2, 1)
sns.histplot(accuracies, kde=True, bins=5)
plt.title(f'Distribuzione dell\'Accuratezza su {k} Iterazioni')
plt.xlabel('Accuratezza')
plt.ylabel('Frequenza')

# Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x=accuracies)
plt.title(f'Boxplot dell\'Accuratezza su {k} Iterazioni')
plt.xlabel('Accuratezza')

plt.tight_layout()
plt.show()

print("\nEsecuzione del progetto terminata.")