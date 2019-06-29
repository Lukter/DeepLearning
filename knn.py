from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# of nearest neighbor for classification")
ap.add_argument("-j", "--jobs", type=int, default=1, help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())

# Step 1 in pipeline

print("[INFO] loading image...")

##Pega o caminho passado na linha de comando
imagePaths = list(paths.list_images(args["dataset"]))

##Cria o preprocessador para dar resize nos data points
sp = SimplePreprocessor(32,32)

##Cria o data loader para ler todos os arquivos
sdl = SimpleDatasetLoader(preprocessors=[sp])

## da load em todos os datapoints
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.reshape((data.shape[0], 3072))

print("[INFO] features matrix: {:.1f}MB".format(data.nbytes/(1024 * 1000.0)))


# Step 2 in pipeline

le = LabelEncoder()

## transforma os labels 'cat' e 'dog' para inteiros. (Mta usado em machine learning)
labels = le.fit_transform(labels)

## Aqui é calculado a quantidade de datapoints para treino do ML e teste do ML

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

print(trainX)
print(testX)
# Steps 3 and 4

print("[INFO] evaluating k-NN classifier...")
model = KNeighborsClassifier(n_neighbors=args["neighbors"], n_jobs=args["jobs"])

model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))


