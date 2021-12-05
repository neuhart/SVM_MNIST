import numpy as np
import csv
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import pandas as pd
import time
from PIL import Image

"Used Classifier"
list_classifier= [svm.SVC(kernel='rbf', gamma='scale'),svm.SVC(kernel='linear',gamma='scale'),svm.SVC(kernel='poly', degree=2,gamma='auto'),neighbors.KNeighborsClassifier(1)]

"Data Import"
"Train Data"

trainname = 'train.csv'
df  = pd.read_csv(trainname, sep = ",")
Examples=df.values[:,1:]
Labels=df.values[:,0]
SortedListofDigits=[[Examples[i] for i in range(len(Labels)) if int(Labels[i])==j] for j in range(10)]#sorting the digits
List_TE_Images=[[] for i in range(11)]
List_TE_Labels=[[] for i in range(11)]

TestExamples=[]
TestLabels=[]
for j in range(10):
    TestExamples+=SortedListofDigits[j][1000:]
    TestLabels+=[j]*len(SortedListofDigits[j][1000:])

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

x = np.arange(3)  # the label locations
width = 0.2  # the width of the bars
fig, ax = plt.subplots()

for k in range(2): #shift
    values = []
    for i in [1, 5, 10]:  # MNIST/i
        for j in range(10): #digits
            BaseTE = SortedListofDigits[j][100:100 + i]
            List_TE_Images[j] = BaseTE.copy()
            List_TE_Labels[j] = [j] * i

            if not k == 0:  # if k==0: baseline, no additional images
                for base_img in BaseTE:
                    for shift in [-2,-1,1, 2]:
                        for axis in range(2):
                            List_TE_Images[j].append(img_shift(base_img, shift, axis))
                    List_TE_Labels[j] += [j]*8
            List_TE_Images[10] += List_TE_Images[j]
            List_TE_Labels[10] += List_TE_Labels[j]

        'Fitting the Classifier on the training data'
        clf = list_classifier[2]
        clf.fit(List_TE_Images[10], List_TE_Labels[10])

        List_TE_Images = [[] for i in range(11)]
        List_TE_Labels = [[] for i in range(11)]

        Predictions = clf.predict(TestExamples)
        falselyclassified = [TestExamples[i] for i in range(len(TestLabels)) if Predictions[i] != TestLabels[i]]
        print(1 - (len(falselyclassified) / len(TestLabels)))
        values.append(1 - (len(falselyclassified) / len(TestLabels)))
    if k == 0:
        lab = 'no shift'
        sgn = -1
    else:
        lab = 'with shift'
        sgn = +1
    ax.bar(x + sgn * width / 2, values, width, label=lab)
    values = []
ax.set_xticks(x)
ax.set_xticklabels(['MNIST/1', 'MNIST/5', 'MNIST/10'])
plt.title('kNN, k=1')
plt.ylabel('Accuracy')
plt.ylim([0,1])
plt.legend()
fig.tight_layout()
plt.savefig('Shift-Acc_Analysis' + str(time.time()) + '.png')
plt.show()

"Shift Visualization"
#plotMNIST(PlotList[10:20],2)
