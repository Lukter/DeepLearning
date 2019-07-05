import numpy as np
'''
------- AND --------
x0   x1   x0&x1
0    0      0
0    1      0
1    0      0
1    1      1
--------------------
------- OR  --------
x0   x1   x0&x1
0    0      0
0    1      1
1    0      1
1    1      1
--------------------
------- XOR --------
x0   x1   x0&x1
0    0      0
0    1      1
1    0      1
1    1      0
--------------------

'''
class Perceptron:
    # N -> Numero de colunas dos feature vector. Dois no meu caso (x0 e x1)
    # Alpha -> Learning rate
    def __init__(self, N, alpha = 0.1):
        #initialize the weight matrix and store the learning rate
        # Inicializa a matriz de pesos atraves da amostragem de uma gaussiana,
        # com media 0 e variancia 1.
        # Possui N + 1 entradas para comportar N entradas e mais o bias
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    # x -> valor da soma dos resultados das matrizes de peso
    def step(self, x):
        # Eh a funcao de ativacao step do perceptron
        return 1 if x > 0 else 0

    # X -> training data
    # y ->
    # epochs -> numero de epocas q o perceptron ira treinar
    def fit(self, X, y, epochs = 10):
        # Adiciona o bias no final 
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x ,target) in zip(X, y):
                p = self.step(np.dot(x, self.W))

                if p != target:
                    error = p - target

                    self.W += -self.alpha * error * x

    def predict(self, X, addBias = True):
        X = np.atleast_2d(X)

        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]
        
        return self.step(np.dot(X, self.W))
