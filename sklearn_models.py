import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm


# Chargement des données d'entraînement dans un objet DataFrame pandas
train_data = pd.read_csv('basket_train.csv')
#print(train_data)


##########
### Etape 1
### Prétraitement des données
############

# Remplacement des données manquantes par la moyenne par colonne
train_data = train_data.fillna (value = train_data.mean())

# On veut s'assurer que le modèle fonctionnera aussi pour de nouveaux patients. On sépare donc les données en un jeu d'entraînement et un jeu de
# validation.
# On sépare les données de manière à avoir 80 % des données dans l'ensemble d'entraînement et 20 % pour l'ensemble de validation.
# On appelera respectivement ces ensembles diabetes_train et diabetes_valid.

basket_train=train_data.sample(frac=0.8,random_state=90)
baket_valid=train_data.drop(basket_train.index)

# On sépare pour chaque ensemble les features (caractéristiques des personnes) et la cible à prédire ("outcome").
# On notera X_train, Y_train et X_valid, Y_valid respectivement les features et les cibles pour l'ensemble d'entraînement et de validation.
X_train = basket_train[basket_train.columns[:-1]]
Y_train = basket_train[basket_train.columns[-1]]

X_valid = baket_valid[basket_train.columns[:-1]]
Y_valid = baket_valid[basket_train.columns[-1]]

## Affichage des tailles de ces datasets
#print(X_train.shape, X_valid.shape)
#print(X_valid.shape, Y_valid.shape)

# On renormalise les données de façon à ce que chaque colonne des datasets d'entraînement et de validation soit de moyenne nulle
# et d'écart type 1

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std

#print(X_train.mean(axis=0))
#print(X_train.std(axis=0))

X_valid = (X_valid - mean)/std


X_train_np = X_train.values
Y_train_np = Y_train.values

X_valid_np = X_valid.values
Y_valid_np = Y_valid.values

# Chargement des données de test

X_test = pd.read_csv('basket_test_data.csv', index_col=0) 


print("X_test")
#print(X_test)

# Remplacement des données manquantes dans les données de test par la moyenne par colonne
X_test = X_test.fillna (value = X_test.mean())

# Normalisation des données de test.

X_test = (X_test - mean)/std




### model abre de decision ### 

from sklearn.tree import DecisionTreeClassifier

def evaluation(Y_pred, Y_valid_np, verbose=True):
   
    TP = 0  # prediction égale à 1 et cible égale à 1 (vrai positif, true positive TP)
    FP = 0  # prediction égale à 1 et cible égale à 0 (faux positif, false positive FP)

    TN = 0  # prediction égale à 0 et cible égale à 0 (vrai négatif, true negative TN)
    FN = 0  # prediction égale à 0 et cible égale à 1 (faux négatif, false negative TN)

    total = 0

    for i in range(Y_pred.shape[0]):

        if (Y_pred[i] == 1 and Y_valid_np[i] == 1):
            TP += 1
        elif (Y_pred[i] == 1 and Y_valid_np[i] == 0):
            FP += 1
        elif (Y_pred[i] == 0 and Y_valid_np[i] == 0):
            TN += 1
        elif (Y_pred[i] == 0 and Y_valid_np[i] == 1):
            FN += 1

        total += 1

    accuracy = (TP + TN) / total

    if verbose:
        print("Vrais positifs : " + str(TP))
        print("Faux positifs : " + str(FP))
        print("Vrais négatifs : " + str(TN))
        print("Faux négatifs : " + str(FN))

        print("Accuracy:" + str(accuracy))

    return accuracy


list_accuracy = []

for crit in tqdm(range(1,10)):
    dtc = DecisionTreeClassifier(max_depth=crit)
    dtc.fit(X_train, Y_train)
    Y_pred = dtc.predict(X_valid)
    list_accuracy.append(evaluation(Y_pred, Y_valid_np))

print(list_accuracy)

# # Traçage de la courbe qui montre l'évolution de l'accuracy en fonction du parametre max_depth
#

x=range(1,10)
y=list_accuracy

plt.plot(x,y)
plt.xlabel("max_depth")
plt.ylabel("accuracy")

plt.legend()
plt.show()


### foret aleatoire ### 

from sklearn.ensemble import RandomForestClassifier

list_accuracy = []

for estimator in tqdm([int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]):
    rfc = RandomForestClassifier(n_estimators=estimator, random_state=0)
    rfc.fit(X_train, Y_train)
    Y_pred = rfc.predict(X_valid)
    list_accuracy.append(evaluation(Y_pred, Y_valid_np))

print(list_accuracy)

# # Traçage de la courbe qui montre l'évolution de l'accuracy en fonction du parametre random_state
#

x=[int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
y=list_accuracy

plt.plot(x,y)
plt.xlabel("n_estimators")
plt.ylabel("accuracy")

plt.legend()
plt.show()

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, Y_train)
Y_pred = rfc.predict(X_valid)
np.savetxt('basket_test_predictions.csv', Y_pred, delimiter=',')


### reseau de neurone ### 


from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]
clf = GridSearchCV(MLPClassifier(), param_grid, cv=3,
                           scoring='accuracy')
clf.fit(X_train, Y_train)


print("Best parameters set found on development set:")
print(clf.best_params_)

clf = MLPClassifier(activation= 'tanh', hidden_layer_sizes= (18,), solver= 'sgd').fit(X_train, Y_train)
Y_pred = clf.predict(X_test)
np.savetxt('basket_test_predictions.csv', Y_pred, delimiter=',')




