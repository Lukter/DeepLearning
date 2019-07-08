from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")

args = vars(ap.parse_args())

# Faz download do dataset inteiro do MNIST
print("[INFO] loading MNIST (full) dataset...")
dataset = datasets.fetch_openml("mnist_784")

# Faz normalizacao dos pixels, ou seja, coloca as intensidades
# dos pixels entre 0 e 1. Lembrar que cada do pixel (RGB) vai de 0-255.
data = dataset.data.astype("float") / 255.0

# Separa o dataset em X e Y de treino e de teste.
# 75% para treino e 25% para teste.
(trainX, testX, trainY, testY) = train_test_split(data, dataset.target, test_size = 0.25)

# No MNIST, as classes (Y) estao em um range de 0-9 (cada label para um digito)
# Porem, eh necessario que esses labels nao sejam inteiro e sim vectores.
# Para isso, eh feito uma codificacao one hot
# [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
# [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
# [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# Transformacao para one-hot
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


# Instancia um objeto Sequential possibilita a
# adicao de layers em um modelo.
model = Sequential()

# Adiciona o primeiro layers ao modelo. Esse layer eh referente ao input da NN
# No caso de cada datapoint do MNIST, a dimensao dele eh 784. 
# Ainda nessa linha, se aprende 256 pesos e utiliza a funcao de ativacao sigmoid
model.add(Dense(256, input_shape=(784,), activation="sigmoid"))

# Adiciona outro layer q aprende 128 pesos e utiliza a funcao de ativacao sigmoid
model.add(Dense(128, activation="sigmoid"))

# Adiciona o ultimo layers (output) com 10 pesos (um para cada digito 0-9)
# Aqui eh utilizado a funcao de ativacao softmax para obter as probabilidades de classe normalizada
# para cada predicao

model.add(Dense(10, activation="softmax"))

print("[INFO] training network...")

# Define q sera utilizado SGD para aprendizagem, utilizando
# uma taxa de aprendizam de 0.01
sgd = SGD(0.01)

# Define que o modelo ira usa como loss function a funcao crossentropy,
# SGD como metodo de aprendizagem
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Faz o treinamento da rede neural
# Os primeiro argumentos eh o dataset e os labels para treinamento
# Validation data -> sao os valores para teste (dataset de test e seus respectivos labels)
# epochs -> numero de epocas que sera treinado
# batch_size -> numero de amostras q sera usada em cada por vez
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=100, batch_size=128)

print("[INFO] evaluating network...")

# Faz a predicao do dataset de teste
predictions = model.predict(testX, batch_size=128)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=[str(x) for x in lb.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label="trian_loss")

plt.plot(np.arange(0, 100), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 100), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 100), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])

