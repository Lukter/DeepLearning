from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
import argparse


'''
Funcao de ativação. Ela ira "ativar" e "fire ON" (output value > 0.5)
ou "fire OFF" (output value <=0.5), tudo baseado em X.
Se essa equacao for plotada, tera um formato em "S"
'''
def sigmoid_activation(x):
    return 1.0/(1 + np.exp(-x))

'''
Utiliza a funcao de ativacao, passando como argumento o dot product
do valor de entrada (xi) e a matriz W (W + b)
'''
def predict(X, W):
    preds = sigmoid_activation(X.dot(W))

    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1
    
    # retorna a predicao
    return preds

ap = argparse.ArgumentParser()

#Numero de epocas 
ap.add_argument("-e", "--epochs", type=float, default=100, help="# of epochs")

#valor de aplha q sera usado (learning rate para gradient descent)
ap.add_argument("-a", "--alpha", type=float, default=0.01, help="learning rate")
args = vars(ap.parse_args())

#Usado para gerar datasets aleatorios
# Foi criado um dataset com 1000 datapoints (n_samples = 1000), 2 categorias
#(centers = 2). Cada datapoint eh um array 2D (n_features = 2). Os labes sao definidos como 0 ou 1.
(X, y) = make_blobs(n_samples=1000, n_features=2, centers=2, cluster_std=1.5, random_state=1)
y = y.reshape((y.shape[0], 1))

#insert a column of 1’s as the last entry in the feature
#matrix -- this little trick allows us to treat the bias
#as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of
# the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.5, random_state=42)

print("[INFO] training...")
W = np.random.rand(X.shape[1], 1)

losses = []
alpha = args["alpha"]
#loop over the desired number of epochs
for epoch in np.arange(0, args["epochs"]):
    # take the dot product between our features ‘X‘ and the weight
    # matrix ‘W‘, then pass this value through our sigmoid activation
    # function, thereby giving us our predictions on the dataset

    preds = sigmoid_activation(trainX.dot(W))

    # now that we have our predictions, we need to determine the
    # ‘error‘, which is the difference between our predictions and
    # the true values
    error = preds - trainY
    loss = np.sum(error ** 2)
    losses.append(loss)

    # the gradient descent update is the dot product between our
    # features and the error of the predictions
    gradient = trainX.T.dot(error)

    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the
    # term "gradient descent" by taking a small step towards a set
    # of "more optimal" parameters
    W += -args["alpha"] * gradient

    # check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1), loss))


# evaluate our model
print("[INFO] evaluating...")

preds = predict(testX, W)
print(classification_report(testY, preds))

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", s=30)
# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, args["epochs"]), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()


