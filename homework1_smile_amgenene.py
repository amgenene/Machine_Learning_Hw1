import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

def fPC (y, yhat):
    return np.mean(y == yhat)

def measureAccuracyOfPredictors (predictors, X, y):
    pixel_intensity = []
    running_list = np.zeros(y.shape)
    for predictor in predictors:
        r1, c1, r2, c2 = predictor
        pixel_intensity = X[:, r1, c1] - X[:, r2, c2]
        pixel_intensity[pixel_intensity > 0] = 1
        pixel_intensity[pixel_intensity < 0] = 0
        running_list += pixel_intensity
    running_list /= len(predictors)
    running_list[running_list > .5] = 1
    running_list[running_list <= .5] = 0
    return fPC(y, running_list)

def stepwiseRegression (trainingFaces, trainingLabels, testingFaces, testingLabels):
    show = True
    if show:
        # Show an arbitrary test image in grayscale
        im = testingFaces[0, :, :]
        im2 = trainingFaces[0, :, :]
        fig,ax = plt.subplots(1)
        ax.imshow(im2, cmap='gray')
        # Show r1,c1
        predictors = []
        # for m in range(0,5):
        current_best_predictors = 0
        for j in range(0, 5):
            current_best_accuracy = 0
            current_best_pixels = 0
            for r1 in range(0,24):
                for c1 in range(0,24):
                    for r2 in range(0,24):
                        for c2 in range(0,24):
                            if r1 == r2 and c1 == c2:
                                continue
                            if (r1, c1, r2, c2) in predictors:
                                continue
                            else:
                                current_accuracy = measureAccuracyOfPredictors(predictors + list(((r1, c1, r2, c2),)),
                                                                               trainingFaces, trainingLabels)
                                if current_accuracy > current_best_accuracy:
                                    current_best_accuracy = current_accuracy
                                    current_best_pixels = (r1, c1, r2, c2)
            predictors.append(current_best_pixels)

        for predictor in predictors:
            print(predictor)
            r1, c1, r2, c2 = predictor
            rect = patches.Rectangle((r1, c1),1,1,linewidth=2,edgecolor='r',facecolor='none')
            ax.add_patch(rect)
            # # Show r2,c2
            rect = patches.Rectangle((r2,c2),1,1,linewidth=2,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
            # # Display the merged result
        plt.show()

        return predictors
'TODO make this train the model using stepwise classification, and run on the test set'
def run_model(trainingFaces, trainingLabels, testingFaces, testingLabels):
    n = [400,800,1200,1600,2000]
    predictors = []
    for j in n:
        predictors = stepwiseRegression(trainingFaces[0:j,:,:], trainingLabels[0:j,:,:], testingFaces,testingLabels)
        print('Predictors: ', predictors)
        accuracy_on_training_set = measureAccuracyOfPredictors(predictors, trainingFaces, trainingLabels)
        print('accuracy_on_training_set: ', accuracy_on_training_set)
        accuracy_on_testing_set = measureAccuracyOfPredictors(predictors, testingFaces, testingLabels)
        print('accuracy_on_testing_set: ', accuracy_on_testing_set)

def loadData (which):
    faces = np.load("{}ingFaces.npy".format(which))
    faces = faces.reshape(-1, 24, 24)  # Reshape from 576 to 24x24
    labels = np.load("{}ingLabels.npy".format(which))
    return faces, labels

if __name__ == "__main__":
    testingFaces, testingLabels = loadData("test")
    trainingFaces, trainingLabels = loadData("train")
    #run_model(trainingFaces, trainingLabels, testingFaces, testingLabels)
    stepwiseRegression(trainingFaces, trainingLabels, testingFaces, testingLabels)
