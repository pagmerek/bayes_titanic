import numpy as np
import pandas as pd
import math

def pre_processor(data):
        """ Podział danych na cechy oraz wektor docelowych wartosci (kolumn Survived) """
        
        x = data.drop('survived',1)
        y = data['survived']
    
        return x,y

def calculate_accuracy(val_predicted, val_true):
    return round(float(sum(val_predicted==val_true))/float(len(val_true))*100,4)

class NBClassifier: 

    def __init__(self):
        self.attributes = list
        self.probabilites = {}
        self.target_priors = {}

        self.x_train = np.array
        self.y_train = np.array
        self.train_size = int
        self.num_feats = int

    def calculate_prior(self):
        """ 
        Zliczenie ilu pasażerów przetrwało a ilu zmarło i na podstawie tych danych wyliczenie prawdopodobieństwa 
        P(survived = 0) oraz P(survived = 1)
        """
        survived_count = sum(self.y_train == 1)
        self.target_priors[1] = survived_count/self.train_size

        died_count = sum(self.y_train == 0)
        self.target_priors[1] = died_count/self.train_size

    def calculate_prob(self):
        """ 
        Obliczenie dla każdej cechy średniej oraz wariancji której użyjemy potem do rozkładu normalnego 
        P(cecha|survived = 0) oraz P(cecha|survived = 1)
        """
        for attribute in self.attributes:
            self.probabilities[attribute][1]['mean'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].mean()
            self.probabilities[attribute][1]['variance'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].var()

            self.probabilities[attribute][0]['mean'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].mean()
            self.probabilities[attribute][0]['variance'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].var()


