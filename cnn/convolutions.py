from skimage.exposure import rescale_intensity
import numpy as np
import argparse
import cv2


def convolve(image, K):
    (iH, iW) = image.shape[:2]
    (kH, kW) = K.shape[:2]

    # Replicando os pixels da borda da imagem para não se perderem no processo de convolucao
    pad = (kW -1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)

    # Inicializando a imagem de saida (output da convolucao)
    output = np.zeros((iH, iW), dtype="float")
    
    # Faz os "sliding" do kernel na imagem
    # O movimento começa pelo y (altura) e faz um loop no x
    # Ex: imagem:
    # 93 , 139, 101,  2
    # 26 , 252, 196,  6
    # 135, 230,  18,  5
    # 1  ,   2,   3,  4
    #
    # k:
    # 1, 2, 3
    # 4, 5, 6
    # 7, 8, 9
    #
    # A matriz de convolucao na posicao (1,1) vai ser:
    # 93 , 139, 101
    # 26 , 252, 196
    # 135, 230, 18
    
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
    
            #Calculo da convolucao
            k = (roi * K).sum()
            output[y - pad, x - pad] = k

    # A convolucao pode fazer com que as intensidades dos pixels passem
    # de 255. Para traze-los de volta para essa faixa, utiliza-se o
    # rescale_intensity
    output = rescale_intensity(output, in_range=(90, 255))

    # Os valores estao em float, mas a imagem original era em 8 bits, portanto
    # faz com que os valores voltem para esse formato.
    output = (output * 255).astype("uint8")

    return output

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
help="path to the input image")
args = vars(ap.parse_args())

# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))



sharpen = np.array(([0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]), dtype="int")


laplacian = np.array(([0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]), dtype="int")

sobelX = np.array(([-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]), dtype="int")

sobelY = np.array(([-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]), dtype="int")

emboss = np.array(([-2, -1, 0],
                   [-1, 1, 1],
                   [0, 1, 2]), dtype="int")

kernelBank = (("small_blur", smallBlur),
              ("large_blur", largeBlur),
              ("sharpen", sharpen),
              ("laplacian", laplacian),
              ("sobel_x", sobelX),
              ("sobel_y", sobelY),
              ("emboss", emboss))

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for (kernelName, K) in kernelBank:
    print("[INFO] applying {} kernel".format(kernelName))
    convolveOutput = convolve(gray, K)
    opencvOutput = cv2.filter2D(gray, -1, K)

#    cv2.imshow("Original", gray)
#    cv2.imshow("{} - convolve".format(kernelName), convolveOutput)
#    cv2.imshow("{} - opencv".format(kernelName), opencvOutput)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
