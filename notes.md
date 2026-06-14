### **DATASET PaHaW**



Dataset containing handwriting time series signals. Note that many features can also be manually engineered. Performance may vary depending on feature design and modeling choices. 



### **CHALLENGE DETAILS**

#### 

#### **Dataset Description**



1. The Parkinson's Disease Handwriting Database (PaHaW) consists of multiple handwriting samples from 37 parkinsonian patients (19 men/18 women) and 38 gender and age matched controls (20 men/18 women). The database was acquired in cooperation with the Movement Disorders Center at the First Department of Neurology, Masaryk University and St. Anne's University Hospital in Brno, Czech Republic.

2. Each subject was asked to complete a handwriting task according to the prepared filled template at a comfortable speed. The completed template was shown to the subjects; no restrictions about the number of repetitions of syllables/words in tasks or their height were given.

3. A tablet was overlaid with a empty paper template (containing only printed lines and square box specifying area for Archimedean spiral), and a conventional ink pen was held in a normal fashion, allowing for immediate full visual feedback. The signals were recorded using the Intuos 4M (Wacom technology) digitizing tablet with 150 Hz sampling frequency.

4. Digitized signals were acquired during the movements executed while exerting pressure on the writing surface and during the movement above the writing surface. We denote these signals as on-surface movement and in-air movement, respectively. The perpendicular pressure exerted on the tablet surface was also recorded. The recordings started when the pen touched the surface of the digitizer and finished when the task was completed. the tablet captured the following dynamic features (time-sequences): x-coordinate; y-coordinate; time stamp; button status; pressure; tilt; and elevation. Button status is a binary variable, being 0 for pen-up state (in-air movement) and 1 for pen-down state (on-surface movement).



#### **SVC file structure**

1st line: number of samples

n-th line: Y coordinate, X coordinate, time stamp, button state, azimuth, altitude, pressure.



#### **Objective**

Build a binary classification model to distinguish Parkinson’s disease patients (1) from healthy subjects (0).



#### **Educational Hint**

Subjects \['00061', '00080', '00089'] did not complete all tasks and can be safely removed a priori.

It is suggested to first experiment with raw time-series classification.

Then try converting time series into images (e.g., recurrence plots, spectrogram-like representations).

However, approaches are not limited to these two strategies; feature engineering and hybrid methods are also encouraged. TRY!



#### **Submission rules**

A CSV file with a header.

One row for each ID in the test set.

The PD status column must contain the predicted class (0 for healthy patients, 1 for patients affected by Parkinson’s disease).



#### **Evaluation**

The official evaluation metric are F1-Score, Precision, Accuracy, Recall, Specificity and AUC.

