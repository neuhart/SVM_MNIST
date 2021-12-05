import numpy as np
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image
import os

def img_shift(img_arr, int_shift, axis):
    if int_shift==0: return img_arr
    newimage = np.roll(np.reshape(img_arr,(28,28)), int_shift, axis=axis)
    if axis == 0:
        if int_shift == 2: newimage[1, :] = np.zeros(28)
        if int_shift >= 1: newimage[0, :] = np.zeros(28)
        if int_shift <= -1: newimage[27, :] = np.zeros(28)
        if int_shift == -2: newimage[26, :] = np.zeros(28)
    if axis == 1:
        if int_shift == 2: newimage[:, 1] = np.zeros(28)
        if int_shift >= 1: newimage[:, 0] = np.zeros(28)
        if int_shift <= -1: newimage[:, 27] = np.zeros(28)
        if int_shift == -2: newimage[:, 26] = np.zeros(28)
    newimage=np.reshape(newimage,784)
    return newimage

def img_rotation(arr_img,deg): #arr_img= input image to be rotated, rotation= degree of rotation
    pre_img=(np.array(arr_img).reshape(28, 28)).astype(np.uint8) #reshaping, etc
    img = Image.fromarray(pre_img, mode='L') #converting to image
    newimage=np.array(img.rotate(deg)).reshape(784) #rotating image and converting back to array
    return newimage

def plotMNIST(Images_l,plot_rows):
    #Images must be a list consisting of 28x28 arrays, plot_rows must be a divider of len(Images)
    fig, axs = plt.subplots(plot_rows, int(len(Images_l)/plot_rows))
    for m in range(plot_rows):
        for i in range(int(len(Images_l)/plot_rows)):
            h_map_x=[]
            h_map_y =[]
            color=[]
            for j in range(28):
                h_map_x= h_map_x + [j]*28
                h_map_y=h_map_y + [l for l in range(28)]
                color= color + [str(1-Images_l[m*int(len(Images_l)/plot_rows)+i][784-((28-j)+k*28)]/255) for k in range(28)]

            axs[m,i].scatter(h_map_x, h_map_y, s = 50, c = color)
    for ax in axs.flat:
        ax.label_outer()
    plt.savefig('Image_plot_6_'+ str(time.time()) + '.png')
    plt.show()
    return 0

def plot_title(clf):
    if clf.__class__.__name__== 'SVC':
        if clf.__class__.get_params(clf)['kernel'] == 'poly':
            return str(clf.__class__.__name__) + ' kernel=' + str(clf.__class__.get_params(clf)['kernel']) + ' deg=' + str(clf.__class__.get_params(clf)['degree'])
        else:
            return str(clf.__class__.__name__)+ ' kernel='+ str(clf.__class__.get_params(clf)['kernel'])
    if clf.__class__.__name__ == 'KNeighborsClassifier':
        return str(clf.__class__.__name__)+ ' k=' + str(clf.__class__.get_params(clf)['n_neighbors'])

def test_classifier(clf,training_observations,training_labels, test_observations, test_labels):
    clf.fit(training_observations,training_labels)
    Predictions = clf.predict(test_observations)
    falselyclassified = [test_observations[i] for i in range(len(test_labels)) if Predictions[i] != test_labels[i]]
    return 1 - (len(falselyclassified) / len(test_labels))

"Used Classifier"
list_classifier= [neighbors.KNeighborsClassifier(1),svm.SVC(kernel='linear',gamma='auto'),svm.SVC(kernel='poly', degree=2,gamma='auto'),svm.SVC(kernel='rbf', gamma='scale')]

"Data Import"
"Training Data"
files=[['./Supertrainers/' + filename for filename in os.listdir('./Supertrainers/')],
['./Supertrainers_5/' + filename for filename in os.listdir('./Supertrainers_5/')],
['./Supertrainers_10/' + filename for filename in os.listdir('./Supertrainers_10/')]]

"Test Data"

trainname = 'train.csv'
df2  = pd.read_csv(trainname, sep = ",")
Examples2=df2.values[:,1:]
Labels2=df2.values[:,0]
SortedListofTestExamples=[[Examples2[i] for i in range(len(Labels2)) if int(Labels2[i])==j] for j in range(10)] #sorting the digits
TestExamples=[]
TestLabels=[]
for j in range(10):
    TestExamples+=SortedListofTestExamples[j][1000:]
    TestLabels+=[j]*len(SortedListofTestExamples[j][1000:])


"mean accuracy values of MNIST/1/5/10 sets for all classifiers"
mean_acc=[[0.42,0.64,0.71],[0.42,0.67,0.76],[0.29,0.61,0.73],[0.42,0.62,0.75]]

for i in range(len(list_classifier)):
    x = np.arange(3)  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - 3 * width / 3, mean_acc[i], width, label='mean')  # add mean value bar for plots

    List_TE_Images = [[] for i in range(11)]
    List_TE_Labels = [[] for i in range(11)]
    BaseTE=[]


    for k in range(2):
        values=[]
        for idx, size_DS in enumerate([1,5,10]):
            "Train Data"
            supertrainers = files[idx][i]
            df = pd.read_csv(supertrainers, sep=",")
            Examples = df.values[:, :]
            Labels = []
            if size_DS==1: Labels = [i for i in range(10)] * 10
            if size_DS == 5:
                Examples=Examples[:50]
                for j in range(10):
                    Labels += [j] * 5
            if size_DS == 10:
                for j in range(10):
                    Labels += [j] * 10
            SortedListofTrainingExamples = [[Examples[i] for i in range(len(Labels)) if int(Labels[i]) == j] for j in
                                            range(10)]  # sorting the digits

            for j in range(10):
                BaseTE = SortedListofTrainingExamples[j][:size_DS]
                List_TE_Images[j] = BaseTE.copy()
                List_TE_Labels[j] = [j] * size_DS

                if not k == 0:  # if k==0: baseline, no additional images
                    for base_img in BaseTE:
                        for shift in [-2,-1,1, 2]:
                            for axis in range(2):
                                List_TE_Images[j].append(img_shift(base_img, shift, axis))
                        List_TE_Labels[j] += [j]*8
                        for angle in range(1,16):
                            List_TE_Images[j].append(img_rotation(base_img, angle))
                            List_TE_Images[j].append(img_rotation(base_img, -angle))
                            List_TE_Labels[j] += [j] * 2
                List_TE_Images[10] += List_TE_Images[j]
                List_TE_Labels[10] += List_TE_Labels[j]

            'Fitting the Classifier on the training data'
            Acc=test_classifier(list_classifier[i],List_TE_Images[10],List_TE_Labels[10],TestExamples,TestLabels)
            List_TE_Images = [[] for i in range(11)]
            List_TE_Labels = [[] for i in range(11)]
            print(Acc)
            values.append(Acc)
        if k == 0:
            lab = 'super-trainers'
            sgn = 0
        else:
            lab = 'incl. shift+rotation'
            sgn = +3
        ax.bar(x + sgn * width / 3, values, width, label=lab)
        values = []

    ax.set_xticks(x)
    ax.set_xticklabels(['MNIST/1', 'MNIST/5', 'MNIST/10'])
    plt.title(plot_title(list_classifier[i]))
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend()
    fig.tight_layout()
    plt.savefig('Shift+Rotation_Analysis ' + plot_title(list_classifier[i]) + '.png')
    plt.show()

