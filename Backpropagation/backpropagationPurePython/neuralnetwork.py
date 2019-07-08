import numpy as np


class NeuralNetwork:
    def __init__(self, layers, alpha = 0.1):
        # Matriz de pesos
        self.W = []

        # camadas da rede eneural (2-2-1, por exemplo)
        self.layers = layers
        
        # taxa de aprendizagem da rede neural
        self.alpha = alpha
#        print('Layers: ' )
#        print(layers)
#        print('Numero de layers: ' + str(len(self.layers)))
        # Inicializa as matrizes de peso (w) da rede
        # Supondo uma rede 2-2-1, temos que inicializar matrizes
        # apenas do primeiro layer. Essa trecho tmb adiciona um
        # um input do bias.
        for i in np.arange(0, len(layers) - 2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

#        print('Pesos gerados: ')
#        print(self.W)

        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

#        print('Pesos finais gerados: ')
#        print(self.W)

    def __repr__(self):
        return "NeuralNetwork: {}".format("-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
 #       print('Input recebido: ')
 #       print('Sigmoid calculado: ')
 #       print(1.0 / (1 + np.exp(-x)))
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
 #       print('Valor recebido:')
 #       print(x)
 #       print('Sigmoid deriv calculado')
 #       print(x * (1 - x))
        return x * (1 - x)

    def fit(self, X, y, epochs = 1000, displayUpdate = 100):
#        print('TREINANDO A REDE')
        X = np.c_[X, np.ones((X.shape[0]))]

        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
 #               print('DATA POINT: ')
#                print(x)
#                print('CLASSE DO DATAPOINT: ')
#                print(target)
                self.fit_partial(x, target)

            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        A = [np.atleast_2d(x)]
 #       print('Valor de A inicial: ')
 #       print(A)

#        print('Iniciando fase de feedfoward')
        #FEEDFOWARD
        for layer in np.arange(0, len(self.W)):
#            print('Layer: ')
#            print(layer)
#            print('Output activation atual: ')
#            print(A[layer])
 #           print('Peso atual: ')
#            print(self.W[layer])
            net = A[layer].dot(self.W[layer])
 #           print('Layers input: ')
#            print(net)
            out = self.sigmoid(net)
 #           print('Layer output: ')
#            print(out)
            A.append(out)
 #           print('A atual: ')
#            print(A)

#        print('FASE DE BACKPROPAGATION')
        #BACKPROPAGATION
        error = A[-1] - y
#        print('Valor de error: ')
#        print(error)
        D = [error * self.sigmoid_deriv(A[-1])]
#        print('Lista de Deltas: ')
#        print(D)

        for layer in np.arange(len(A) - 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        D = D[::-1]

        for layer in np.arange(0, len(self.W)):

            self.W[layer] += -self.alpha*A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        p = np.atleast_2d(X)

        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        for layers in np.arange(0, len(self.W)):

            p = self.sigmoid(np.dot(p, self.W[layers]))

        return p

    def calculate_loss(self, X, targets):
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
