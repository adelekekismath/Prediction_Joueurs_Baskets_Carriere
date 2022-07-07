import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
import pandas as pd
from tqdm import tqdm


# Chargement des données d'entraînement dans un objet DataFrame pandas
train_data = pd.read_csv('basket_train.csv')
print(train_data)

# Affichage du descriptif des données
print(train_data.describe())

# Visualisation des données - projection selon les axes "Glucose" et "BMI"
## Dans ce dataset, il y a deux classes 1 ou 0, pour la présence ou l'absence de diabète.




##########
### Etape 2
### Prétraitement des données
############

# Remplacement des données manquantes par la moyenne par colonne
train_data = train_data.fillna (value = train_data.mean())

# On veut s'assurer que le modèle fonctionnera aussi pour de nouveaux patients. On sépare donc les données en un jeu d'entraînement et un jeu de
# validation

# On sépare les données de manière à avoir 80 % des données dans l'ensemble d'entraînement et 20 % pour l'ensemble de validation.
# On appelera respectivement ces ensembles basket_train et basket_valid.

basket_train=train_data.sample(frac=0.8,random_state=90)
basket_valid=train_data.drop(basket_train.index)

# On Sépare pour chaque ensemble les features (caractéristiques des personnes) et la cible à prédire ("outcome").
# On notera X_train, Y_train et X_valid, Y_valid respectivement les features et les cibles pour l'ensemble d'entraînement et de validation.
X_train = basket_train[basket_train.columns[:-1]]
Y_train = basket_train[basket_train.columns[-1]]

X_valid = basket_valid[basket_train.columns[:-1]]
Y_valid = basket_valid[basket_train.columns[-1]]

## Affichage des tailles de ces datasets
print(X_train.shape, X_valid.shape)
print(X_valid.shape, Y_valid.shape)


# On renormalise les données de façon à ce que chaque colonne des datasets d'entraînement et de validation soit de moyenne nulle
# et d'écart type 1

mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

X_train = (X_train - mean)/std

print(X_train.mean(axis=0))
print(X_train.std(axis=0))

X_valid = (X_valid - mean)/std

print(X_valid.mean(axis=0))
print(X_valid.std(axis=0))

### Conversion des DataFrame pandas en en numpy array.
X_train_np = X_train.values
Y_train_np = Y_train.values

X_valid_np = X_valid.values
Y_valid_np = Y_valid.values

## Calcul du nombre de features (nombre de colonne de la matrice X)
d = X_train_np.shape[1]

##  ajout d'une première colonne de "1" aux matrices des entrées X_train et X_valid
X_train_np = np.hstack((np.ones((X_train_np.shape[0],1)),X_train_np))
X_valid_np = np.hstack((np.ones((X_valid_np.shape[0],1)),X_valid_np))


############
### Etape 3
### Algorithme de régression logistique - construction du modèle
############


## Création d'un vecteur de poids de taille d+1 où d est le nombre de features.
weights = np.random.randn(d+1)

### Premier test : calcul du produit scalaire entre la première entrée du dataset et le vecteur de poids
first_entry = X_train_np[0,:]
scalar_product = (weights * first_entry).sum()
print(scalar_product)

#Fonction sigmoid
def sigmoid(z):
    return 1 / (1+np.exp(-z))

# Calcul de la sortie du modèle pour la première entrée
f = sigmoid(scalar_product)
print("f " + str(f))

# Calcul de la prédiction faite par le modèle pour la première entrée
pred = f.round()
print("pred " + str(pred))

# Fonction qui calcule la sortie du modèle pour une matrice de points en entrée
# Version naive avec une boucle


def output(X,weights):
    N = X.shape[0]
    out = np.zeros((N,1))
    for i in range(N):
        out[i] = sigmoid((weights * X[i,:]).sum())
    return out

print(output(X_train_np,weights))

# Version avec un calcul matriciel
'''def output(X,weights):
    return sigmoid(np.dot(X,weights))

'''
# fonction qui calcule les prédictions (0 ou 1) à partir des sorties du modèle
def prediction(f):
    return f.round()


############
### Etape 4
### Algorithme du perceptron - apprentissage du modèle
############

# Calcul de la binary cross entropy entre le vecteur de sortie du modèle et le vecteur des targets (voir cours).
def binary_cross_entropy(f,y):
    return - (y*np.log(f)+ (1-y)*np.log(1-f)).mean()

# Calcul du gradient de l'erreur par rapport aux paramètres du modèle
# a) Version avec une boucle
def gradient(f,y,X):

    grad = np.zeros((d+1))

    for j in range(0,d+1):

        grad[j] = -((y-f)*X[:,j]).mean()

    return grad

#  Version avec un calcul matriciel
# def gradient_dot(f,y,X):
#
#     grad = -np.dot(np.transpose(X),(y-f))/X.shape[0]
#
# #     return grad


# Taux d'apprentissage (learning rate)
eta = 0.002

# Fonction qui calcule le taux d'erreur en comparant le y prédit avec le y réel
def error_rate(Y_pred,Y_test):

    return (np.abs(Y_pred - Y_test).sum())/Y_pred.shape[0]


# Apprentissage du modèle et calcul de la performance tous les 100 itérations
'''nb_epochs = 10000
for i in range(nb_epochs):

    f_train = output(X_train_np,weights)
    Y_pred_train = prediction(f_train)

    grad = gradient(f_train,Y_train_np,X_train_np)

    weights = weights - eta*grad

    if(i%100==0):

        error_train = error_rate(Y_pred_train,Y_train_np)
        loss = binary_cross_entropy(f_train,Y_pred_train)

        f_test = output(X_valid_np, weights)
        y_pred_test = prediction(f_test)

        error_test = error_rate(y_pred_test, Y_valid_np)

        print("iter : " + str(i) +  " error train : " + str(error_train) + " loss " + str(loss) + " error test : " + str(error_test))
'''

def get_weights(eta):
    weights = np.random.randn(d+1)
    for i in range(10000):

        f_train = output(X_train_np,weights)
        Y_pred_train = prediction(f_train)

        grad = gradient(f_train,Y_train_np,X_train_np)

        weights = weights - eta*grad

        if(i%100==0):

            error_train = error_rate(Y_pred_train,Y_train_np)
            loss = binary_cross_entropy(f_train,Y_pred_train)

            f_test = output(X_valid_np, weights)
            y_pred_test = prediction(f_test)

            error_test = error_rate(y_pred_test, Y_valid_np)

            print("iter : " + str(i) +  " error train : " + str(error_train) + " loss " + str(loss) + " error test : " + str(error_test))

    return weights



# Affichage des paramètres appris du modèle
print("weights")
weights = [ 1.54136554 , 1.08147464 , 0.87742655 , 1.61159503 , 0.4381262  , 0.95686491,
  0.03374307, -0.04612172,  2.56258128 , 1.26460447 , 1.89019757 , 0.81485374,
 -0.38308371, -0.24609458,  0.75201876 , 0.46365888]
print(weights)



# Fonction d'evaluation qui permet de calculer et d'afficher le nombre de vrais positifs, de faux positifs, de vrais négatifs et
# de faux négatifs pour un ensemble de validation donné en entrée. La fonction evaluation calcule également la métrique "accuracy" (précision), qui correspond
# à la proportion de prédictions correctes.


def evaluation(X, Y, weights, verbose=True):

    TP = 0  # prediction égale à 1 et cible égale à 1 (vrai positif)
    FP = 0  # prediction égale à 1 et cible égale à 0 (faux positif)

    TN = 0  # prediction égale à 0 et cible égale à 0 (vrai négatif)
    FN = 0  # prediction égale à 0 et cible égale à 1 (faux négatif)

    total = 0
    

    Y_pred = prediction(output(X, weights))

    for i in range(Y_pred.shape[0]):

        if (Y_pred[i] == 1 and Y[i] == 1):
            TP += 1
        elif (Y_pred[i] == 1 and Y[i] == 0):
            FP += 1
        elif (Y_pred[i] == 0 and Y[i] == 0):
            TN += 1
        elif (Y_pred[i] == 0 and Y[i] == 1):
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

## Test de cette fontion d'évaluation
#evaluation(X_valid_np, Y_valid_np, weights)

list_accuracy = []

for eta in tqdm(np.arange(0.001,0.019,0.001)):
    weights= get_weights(eta)
    list_accuracy.append(evaluation(X_valid_np, Y_valid_np, weights))

print(list_accuracy)

# # Traçage de la courbe qui montre l'évolution de l'accuracy en fonction du taux d'apprentissage
#

x=np.arange(0.001,0.019,0.001)
y=list_accuracy

plt.plot(x,y)
plt.xlabel("learning rate")
plt.ylabel("score")

plt.legend()
plt.show()



############
### Etape 5
###  Application du modèle sur des données de test
############

# Chargement des données de test
X_test = pd.read_csv('basket_test_data.csv', index_col=0)

print("X_test")
print(X_test)



# Remplacement des données manquantes dans les données de test par la moyenne par colonne
X_test = X_test.fillna (value = X_test.mean())

# Normalisation des données de test.
X_test = (X_test - mean)/std

### Conversion des DataFrame pandas en en numpy array.
X_test_np = X_test.values
#Y_test_np = Y_test.values
X_test_np = np.hstack((np.ones((X_test_np.shape[0],1)),X_test_np))
##  ajout d'une première colonne de "1" aux matrices des entrées X_test


def get_prediction_file( X_test, weights):
    weights= get_weights(0.017)
    Y_pred = prediction(output(X_test, weights))
    
    np.savetxt('basket_test_predictions.csv', Y_pred, delimiter=',')
        

#get_prediction_file( X_test_np, weights)

'''from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train_np, Y_train_np)
y_pred = reg.predict(X_test_np)
np.savetxt('basket_test_predictions.csv', y_pred, delimiter=',')



#### Test avec la librairie scikit-learn
# Lancement du modèle avec la librairie scikit-learn et affichage des résultats
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train_np,Y_train_np)

Y_pred_valid= model.predict(X_valid_np)
error_valid = error_rate(Y_pred_valid, Y_valid_np)

print("error_valid")
print(error_valid)

print("Valeurs des poids du model avec scikitlearn")
print(model.intercept_, model.coef_)

Y_pred_test= model.predict(X_test_np)
error_test = error_rate(Y_pred_test, Y_test_np)

print("error_test")
print(error_test)'''




############
### Etape 6
### Apprentissage et visualisation du modèle dans le cas où il n'y a que les variables Nb matchs et Points moyens marques  (ce qui permet de faire une visualisation en 2D).
############

### Extraction des sous dataframe avec seulement les variables BMI et Glucose.
X_train = X_train[["Nb matchs","Points moyens marques"]]
X_valid = X_valid[["Nb matchs","Points moyens marques"]]

### Conversion des DataFrame pandas en en numpy array.
X_train_np = X_train.values
Y_train_np = Y_train.values

X_valid_np = X_valid.values
Y_valid_np = Y_valid.values

## Calcul du nombre de features (nombre de colonnes de la matrice X)
d = X_train_np.shape[1]

## Ajout d'une première colonne de "1" aux matrices des entrées X_train et X_valid

X_train_np = np.hstack((np.ones((X_train_np.shape[0],1)),X_train_np))
X_valid_np = np.hstack((np.ones((X_valid_np.shape[0],1)),X_valid_np))

## Création d'un vecteur de poids de taille d+1 où d est le nombre de features.
weights = np.random.randn(d+1)

# Affichage d'une carte des décisions prises par le modèle par rapport aux variables
cmap_light = ListedColormap(['#FFAAAA','#AAFFAA'])

def plot_decision(X,Y,weights):

    h = 0.1

    minX = np.min(X)
    maxX = np.max(X)

    xx1, xx2 = np.meshgrid(np.arange(minX,maxX,h),np.arange(minX,maxX,h))

    # print(xx1)
    # print(xx2)

    xx1_flat = xx1.reshape(xx1.shape[0]**2,1)
    xx2_flat = xx2.reshape(xx2.shape[0]**2,1)


    X_entry = np.hstack((np.ones((xx1_flat.shape[0],1)),xx1_flat,xx2_flat))

    print("X_entry.shape")
    print(X_entry.shape)

    f = output(X_entry,weights)
    Y_pred = prediction(f)

    yy = Y_pred.reshape(xx1.shape[0],xx1.shape[1])

    plt.pcolormesh(xx1,xx2,yy,cmap=cmap_light)

    plt.scatter(X[Y==0,1],X[Y==0,2],color="r")
    plt.scatter(X[Y==1,1],X[Y==1,2],color="g")

    plt.show()


## Affichage de la frontière de décision du modèle avant l'apprentissage
plot_decision(X_valid_np,Y_valid_np,weights)

