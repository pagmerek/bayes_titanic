import numpy as np
import pandas as pd
import math

def pre_processor(data):
    """ Podział danych na cechy oraz wektor docelowych wartosci (kolumn Survived) """

    x = data.drop('survived', 1)
    y = data['survived']

    return x, y


def calculate_accuracy(val_predicted, val_true):
    return round(float(sum(val_predicted == val_true))/float(len(val_true))*100, 4)


def normal_distribution(var, mean, value):
    return (1/math.sqrt(2*math.pi*var))*math.exp(-(value-mean)**2/(2*var))


class NBClassifier:
    def __init__(self):
        self.attributes = list()
        self.probabilites = {}
        self.target_priors = dict()

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
        self.target_priors['survived'] = survived_count/self.train_size

        died_count = sum(self.y_train == 0)
        self.target_priors['died'] = died_count/self.train_size

    def calculate_prob(self):
        """ 
        Obliczenie dla każdej cechy średniej oraz wariancji której użyjemy potem do rozkładu normalnego 
        P(cecha|survived = 0) oraz P(cecha|survived = 1)
        """
        for attribute in self.attributes:
            self.probabilites[attribute]['survived']['mean'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].mean()
            self.probabilites[attribute]['survived']['variance'] = self.x_train[attribute][self.y_train[self.y_train == 1].index.values.tolist()].var()

            self.probabilites[attribute]['died']['mean'] = self.x_train[attribute][self.y_train[self.y_train == 0].index.values.tolist()].mean()
            self.probabilites[attribute]['died']['variance'] = self.x_train[attribute][self.y_train[self.y_train == 0].index.values.tolist()].var()

    def train(self, x, y):
        """
        Inicjalizacja wszystkich wartości a następnie obliczenie poszczególnych prawdopodobieństw
        """
        self.attributes = list(x.columns)
        self.x_train = x
        self.y_train = y
        self.train_size = x.shape[0]
        for attribute in self.attributes:
            self.probabilites[attribute] = { 'survived': {}, 'died':{}}
        self.calculate_prob()
        self.calculate_prior()

    def classify(self, x):
        """
        Obliczenie wartości 
        P(survived = 1 | X.attributes) oraz P(survived = 0 | X.attributes)
        oraz wybranie tej większej
        x - zbiór atrybutów do których musimy dobrać wartość pola survived
        """
        attributes, true_survived_values  = pre_processor(x)
        predicted = []
        for _, record in attributes.iterrows():
            survived_prob = self.target_priors['survived']
            died_prob = self.target_priors['died']
            for attribute, value in record.items():
                survi_mean = self.probabilites[attribute]['survived']['mean']
                survi_var = self.probabilites[attribute]['survived']['variance']
                survived_prob *= normal_distribution(survi_var,survi_mean,value)
                
                died_mean = self.probabilites[attribute]['died']['mean']
                died_var = self.probabilites[attribute]['died']['variance']
                died_prob *= normal_distribution(died_var,died_mean,value)
            if died_prob < survived_prob:
                predicted.append(1)
            else:
                predicted.append(0)
            
        return predicted, true_survived_values

