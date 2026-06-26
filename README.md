# Parkinson's Classification Project - Extended 20-Minute Presentation Script & Notes

*Use this expanded structure to fill a 20-minute presentation. A 20-minute speech requires about 2000-2500 words and deep technical/clinical context.*

## 1. Introduction & Clinical Background (Approx. 3-4 mins)
- **The Hook**: Early detection of Parkinson's Disease (PD) is critical for effective management. PD is a neurodegenerative disorder that primarily affects motor control, leading to symptoms like resting tremors, rigidity, and bradykinesia (slowness of movement).
- **The Rationale (Why Handwriting?)**: Handwriting is a complex cognitive and motor task. Long before visible macro-tremors appear, patients often develop subtle micro-tremors and hesitations in their fine motor control. This phenomenon, along with "Micrographia" (abnormally small handwriting), makes digitized handwriting analysis an excellent non-invasive biomarker for early PD detection.
- **The PaHaW Dataset**: We utilized the Parkinson's Disease Handwriting Database. Unlike a simple scanned image of text on paper, this dataset uses a digitizing tablet to capture temporal data at a high sampling rate. We have a rich time-series containing X and Y spatial coordinates, Pen Pressure, Altitude and Azimuth (pen tilt/angle), and Button State (whether the pen is touching the surface or hovering).

## 2. Feature Engineering & Mathematical Approach (Approx. 5 mins)
- **The Problem with Raw Time-Series**: Raw temporal sequences are highly complex. Feeding them directly into Deep Learning models (like RNNs or LSTMs) requires thousands of samples to mathematically converge without memorizing the training data. Our dataset is strictly limited to 72 viable subjects. 
- **The Solution (Global Statistical Features)**: To combat the "Curse of Dimensionality", we transformed the complex time-series into a flat, tabular format via manual feature engineering. 
- **In-Depth Look at the Features**:
  - **Spatial & Pressure Stats**: We computed the mean, standard deviation, minimum, maximum, and median for X, Y, and Pressure. The standard deviation captures the variance and spread of the writing, which is often diminished in PD (Micrographia).
  - **Kinematic Features (Velocity)**: We mathematically derived velocity using the spatial displacement over time: $v = \sqrt{dx^2 + dy^2} / dt$. A lower mean velocity directly correlates to clinical bradykinesia.
  - **Behavioral Features (Pen Lifts)**: By counting the transitions of the 'Button State' from 1 to 0, we calculated the number of pen lifts. PD patients tend to lift the pen more frequently due to hesitations and lack of fluid motor planning.
- **Q: Why manual feature extraction instead of Deep Learning?**
  - **A**: Deep learning is a "black box" that requires massive datasets. With only 72 subjects, a neural network would heavily overfit, learning the specific quirks of these 72 people rather than the actual disease. Hand-crafted features enforce a strong mathematical prior, giving classic ML models a clean, interpretable baseline.

## 3. Data Splitting & Rigorous Validation Strategy (Approx. 4 mins)
- **Train/Test Split**: We split the 72 subjects using an 80/20 ratio. 80% (57 subjects) for training and 20% (15 subjects) as a completely untouched hold-out test set to simulate real-world unseen patients.
- **The Importance of Stratification**: We used a *stratified* split. This guarantees that the percentage of Healthy vs PD patients remains perfectly identical in both the training and testing sets, avoiding skewed evaluations.
- **Stratified 5-Fold Cross-Validation**: On the 57 training samples, we performed a 5-Fold Cross-Validation. We divide the training set into 5 chunks, train on 4, and validate on 1, rotating this process 5 times. 
- **Q: Why is 5-Fold CV absolutely necessary here?**
  - **A**: In small datasets, a single train/validation split has high variance (you might randomly get an "easy" validation set and think your model is perfect). Cross-validation proves that the model's performance is stable and reliable across different data permutations, proving it can truly generalize.

## 4. Model Selection: The Random Forest Classifier (Approx. 4-5 mins)
- **What is a Random Forest?**: It is an ensemble learning method. Instead of relying on one Decision Tree (which is highly prone to overfitting), it builds hundreds of slightly different trees using "Bagging" (Bootstrap Aggregating) and random feature subsets, then averages their predictions.
- **Hyperparameter Optimization**: 
  - `max_depth=7`: We strictly limited how deep the trees could grow to prevent them from memorizing the noise in our small dataset.
  - `min_samples_split=5`: A node must have at least 5 samples to split further.
  - `class_weight='balanced'`: Automatically adjusts weights inversely proportional to class frequencies to handle any mild imbalances.
- **Q: Why Random Forest and not Support Vector Machines (SVM) or Logistic Regression?**
  - **A**: Logistic Regression assumes linear decision boundaries, which is too simplistic for complex kinematic data. SVMs are a great alternative and handle high-dimensional spaces well. However, Random Forests provide an invaluable feature for medical Machine Learning: **Feature Importances** (calculated via Gini impurity decrease). Random Forests allow us to look inside the "black box" and tell the doctors exactly *which* features (e.g., velocity variance or spatial spread) were the most critical in diagnosing the patient.

## 5. Results, Metrics Interpretation & Future Work (Approx. 3 mins)
- **The Metrics**: The baseline model achieved an Accuracy of 66.7% and an AUC (Area Under the ROC Curve) of 0.71 on the 15-patient hold-out test set.
- **Understanding AUC & Recall**: AUC measures the model's ability to separate the two classes across all probability thresholds. In medical screening, we generally prioritize **Recall (Sensitivity)**—we want to catch as many true PD patients as possible, even if it means raising the False Positive rate slightly (lowering Precision).
- **Future Work (The Computer Vision Approach)**:
  - **The Limitation**: Global statistical aggregates (mean, max) "average out" the data over the whole writing task. If a patient has a micro-tremor that lasts for only 0.5 seconds, the global mean velocity won't capture it.
  - **The Next Evolution**: Since this is a Computer Vision project, the next logical step is to convert the 1D time-series into 2D images. We can generate **Spectrograms** (analyzing frequency over time using Short-Time Fourier Transforms) or Kinematic plots. Once converted into images, we can utilize Transfer Learning with pre-trained Convolutional Neural Networks (CNNs like ResNet or VGG) to automatically learn and extract the most subtle spatio-temporal tremor patterns.

--------------

# Progetto Classificazione Parkinson - Script Esteso per Presentazione (20 Minuti)

*Usa questa struttura espansa per coprire 20 minuti di esposizione. Un discorso di 20 minuti richiede circa 2000 parole e un profondo contesto tecnico/clinico.*

## 1. Introduzione e Contesto Clinico (Circa 3-4 min)
- **L'Incipit**: La diagnosi precoce del morbo di Parkinson (PD) è fondamentale. È una malattia neurodegenerativa che colpisce il controllo motorio, portando a tremori a riposo, rigidità e bradicinesia (lentezza nei movimenti).
- **Perché la Scrittura Manuale?**: Scrivere è un compito motorio e cognitivo molto complesso. Molto prima che compaiano i tremori macroscopici visibili, i pazienti sviluppano spesso micro-tremori ed esitazioni nel controllo motorio fine. Questo, unito alla "Micrografia" (la tendenza a scrivere sempre più piccolo), rende l'analisi della scrittura digitalizzata un eccellente biomarcatore non invasivo per la diagnosi precoce.
- **Il Dataset PaHaW**: Non stiamo analizzando una semplice foto del testo scritto, ma dati temporali acquisiti tramite una tavoletta grafica. Abbiamo a disposizione una serie storica ricca che campiona nel tempo le coordinate spaziali X e Y, la pressione della penna, l'altitudine e l'azimut (inclinazione della penna) e lo stato del pulsante (se la penna tocca il foglio o è sollevata).

## 2. Feature Engineering e Approccio Matematico (Circa 5 min)
- **Il Problema delle Serie Temporali Grezze**: Le sequenze temporali sono matematicamente complesse. Darle in pasto direttamente a reti neurali (come RNN o LSTM) richiede migliaia di campioni per evitare che il modello impari a memoria i dati. Il nostro dataset è limitato a soli 72 soggetti.
- **La Soluzione (Statistiche Globali)**: Per combattere la "Maledizione della Dimensionalità", abbiamo trasformato le complesse serie temporali in un formato tabellare piatto tramite la "Feature Engineering" manuale.
- **Analisi delle Feature Estratte**:
  - **Statistiche Spaziali e di Pressione**: Abbiamo calcolato media, deviazione standard, min, max e mediana. La deviazione standard spaziale cattura l'ampiezza della scrittura, che nei pazienti PD spesso diminuisce.
  - **Feature Cinematiche (Velocità)**: Abbiamo derivato matematicamente la velocità usando lo spostamento spaziale nel tempo: $v = \sqrt{dx^2 + dy^2} / dt$. Una velocità media inferiore è il correlato matematico della bradicinesia clinica.
  - **Feature Comportamentali (Sollevamenti penna)**: Contando le transizioni di stato del pulsante da 1 a 0, abbiamo misurato quante volte la penna viene sollevata. I pazienti PD tendono a sollevare la penna molto più spesso a causa di esitazioni.
- **Domanda: Perché l'estrazione manuale invece del Deep Learning?**
  - **Risposta**: Il Deep Learning è una "scatola nera" avida di dati. Con 72 soggetti, una rete neurale andrebbe in un grave overfitting, imparando le caratteristiche specifiche di queste 72 persone e non della malattia. Le feature manuali creano un "prior" matematico forte, offrendo un punto di partenza (baseline) pulito e interpretabile.

## 3. Suddivisione dei Dati e Validazione Rigorosa (Circa 4 min)
- **Train/Test Split**: Abbiamo diviso i 72 soggetti con un rapporto 80/20. L'80% (57 soggetti) per il training e il 20% (15 soggetti) come set di test "hold-out" totalmente intatto, per simulare pazienti reali mai visti dal modello.
- **L'Importanza della Stratificazione**: Uno split "stratificato" garantisce che la proporzione percentuale tra pazienti Sani e Malati rimanga perfettamente identica in entrambi i set, evitando valutazioni distorte.
- **5-Fold Cross-Validation Stratificata**: Sui 57 campioni di addestramento abbiamo applicato la convalida incrociata a 5 fold. Dividiamo il training set in 5 parti: addestriamo su 4 e validiamo sulla rimanente, ruotando per 5 volte.
- **Domanda: Perché la 5-Fold CV è assolutamente necessaria qui?**
  - **Risposta**: Nei dataset piccoli, un singolo split ha un'alta varianza (potresti essere fortunato e ottenere un set di validazione "facile"). La Cross-Validation dimostra che la performance del modello è stabile su diverse permutazioni dei dati, provando che è in grado di generalizzare veramente.

## 4. Selezione del Modello: Il Random Forest Classifier (Circa 4-5 min)
- **Cos'è un Random Forest?**: È un metodo di apprendimento "Ensemble". Invece di affidarsi a un singolo Albero Decisionale (che tende a fare overfitting), costruisce centinaia di alberi leggermente diversi tramite "Bagging" e sottoinsiemi casuali di feature, facendo poi la media delle loro previsioni.
- **Ottimizzazione degli Iperparametri**:
  - `max_depth=7`: Abbiamo limitato la profondità degli alberi per impedire che memorizzassero il rumore presente nei dati.
  - `min_samples_split=5`: Un nodo deve avere almeno 5 campioni per potersi biforcare.
  - `class_weight='balanced'`: Aggiusta automaticamente i pesi in modo inversamente proporzionale alla frequenza delle classi per bilanciare il modello.
- **Domanda: Perché Random Forest e non Support Vector Machines (SVM) o Regressione Logistica?**
  - **Risposta**: La Regressione Logistica presume confini decisionali lineari, troppo semplicistici per dati cinematici complessi. Le SVM sono ottime, ma i Random Forest offrono una funzione inestimabile nel Machine Learning medico: le **Feature Importances** (calcolate tramite la diminuzione dell'impurità di Gini). Questo ci permette di guardare dentro la "scatola nera" e dire ai medici esattamente *quali* parametri (es. la varianza della velocità) sono stati critici per la diagnosi.

## 5. Risultati, Interpretazione e Sviluppi Futuri (Circa 3 min)
- **Le Metriche**: Il modello baseline ha ottenuto un'Accuratezza del 66.7% e un'AUC (Area sotto la Curva ROC) di 0.71 sul test set da 15 pazienti.
- **Capire l'AUC e la Recall**: L'AUC misura la capacità del modello di separare le classi a vari livelli di soglia. Nello screening medico, spesso si privilegia la **Recall (Sensibilità)**: vogliamo intercettare il maggior numero possibile di veri malati PD, anche a costo di alzare leggermente i Falsi Positivi (abbassando la Precisione).
- **Sviluppi Futuri (L'approccio Computer Vision)**:
  - **Il Limite**: Gli aggregati statistici (media, max) "spalmano" il dato sull'intera durata della scrittura. Se un paziente ha un micro-tremore di soli 0.5 secondi, la media globale non lo catturerà.
  - **Il Prossimo Passo**: Essendo questo un progetto di Computer Vision, lo step successivo è convertire le serie temporali 1D in immagini 2D. Possiamo generare **Spettrogrammi** (analizzando la frequenza nel tempo tramite trasformata di Fourier a breve termine) o immagini cinematiche. Una volta trasformati in immagini, possiamo usare Reti Neurali Convoluzionali pre-addestrate (CNN come ResNet) tramite *Transfer Learning* per estrarre automaticamente i pattern di tremore spazio-temporali più sottili.
