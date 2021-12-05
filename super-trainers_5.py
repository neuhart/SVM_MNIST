import numpy as np
import random
from sklearn import svm
from sklearn import neighbors
import matplotlib.pyplot as plt
import pandas as pd
import time
import csv

"Used Classifier"
list_classifier= [svm.SVC(kernel='rbf', gamma='scale'),svm.SVC(kernel='linear', gamma='scale'),svm.SVC(kernel='poly', degree=2,gamma='auto'),neighbors.KNeighborsClassifier(1)]

"Data Import"
"Train Data"
trainname = 'train.csv'

df  = pd.read_csv(trainname, sep = ",")
Examples=df.values[:,1:]
Labels=df.values[:,0]
SortedListofDigits=[[Examples[i] for i in range(len(Labels)) if int(Labels[i])==j] for j in range(10)] #sorting the digits

SortedTrainingExamples=[]
SortedTrainingLabels=[]
TestExamples=[]
TestLabels=[]
for j in range(10):
    TestExamples+=SortedListofDigits[j][1000:]
    TestLabels+=[j]*len(SortedListofDigits[j][1000:])
acc_results=[]
indexes_list=[[] for j in range(10)]

###########################################
'Definition of functions'
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

def filename_csv(clf):
    if clf.__class__.__name__== 'SVC':
        if clf.__class__.get_params(clf)['kernel'] == 'poly':
            return 'Supertrainers_5_' + str(clf.__class__.__name__) + '_kernel=' + str(clf.__class__.get_params(clf)['kernel']) + '_deg=' + str(clf.__class__.get_params(clf)['degree']) + '_mean=' + str(mean_acc)+ '_var=' + str(var_acc) + '.csv'
        else:
            return 'Supertrainers_5_'+ str(clf.__class__.__name__)+ '_kernel='+ str(clf.__class__.get_params(clf)['kernel'])+ '_mean=' +str(mean_acc)+ '_var=' + str(var_acc)  +'.csv'
    if clf.__class__.__name__ == 'KNeighborsClassifier':
        return 'Supertrainers_5_'+ str(clf.__class__.__name__)+ '_k=' +str(clf.__class__.get_params(clf)['n_neighbors']) + '_mean=' + str(mean_acc)+ '_var=' + str(var_acc)  + '.csv'

def takeFirst(elem):
    return elem[0]

def plot_title(clf):
    if clf.__class__.__name__== 'SVC':
        if clf.__class__.get_params(clf)['kernel'] == 'poly':
            return str(clf.__class__.__name__) + ' kernel=' + str(clf.__class__.get_params(clf)['kernel']) + ' deg=' + str(clf.__class__.get_params(clf)['degree'])
        else:
            return str(clf.__class__.__name__)+ ' kernel='+ str(clf.__class__.get_params(clf)['kernel'])
    if clf.__class__.__name__ == 'KNeighborsClassifier':
        return str(clf.__class__.__name__)+ ' k=' + str(clf.__class__.get_params(clf)['n_neighbors'])

def slf_boxplot(inputdata, clf):
    plt.boxplot(inputdata,showfliers = True)
    plt.title(plot_title(clf))
    plt.ylabel('Accuracy')
    plt.savefig('Boxplot ' + filename_csv(clf) + '.png')
    plt.show()

def test_classifier(clf,training_observations,training_labels, test_observations, test_labels):
    clf.fit(training_observations,training_labels)
    Predictions = clf.predict(test_observations)
    falselyclassified = [test_observations[i] for i in range(len(test_labels)) if Predictions[i] != test_labels[i]]
    return 1 - (len(falselyclassified) / len(test_labels))


###########################################

for clf in list_classifier:
    for i in range(500):
        for j in range(10):
            keylist=[(SortedListofDigits[j][key],key) for key in range(1000)]
            #add keynumber for identifying the image later on
            rd_choice= random.choices(keylist,k=5) #choose k random (image,key number)
            indexes_list[j]+=[img_key_pair[1] for img_key_pair in rd_choice] #store image keys
            SortedTrainingExamples += [img_key_pair[0] for img_key_pair in rd_choice] #add image to training examples
            SortedTrainingLabels+=[j]*5

        'Fitting the Classifier on the training data'
        Acc = test_classifier(clf, SortedTrainingExamples, SortedTrainingLabels, TestExamples, TestLabels)
        acc_results.append((Acc, i))
        # adding (accuracy result, i(index of MNIST/5 data sets))
        SortedTrainingExamples = []
        SortedTrainingLabels = []

    mean_acc=np.mean(np.array(acc_results)[:,0])
    var_acc=np.var(np.array(acc_results)[:,0])
    slf_boxplot(np.array(acc_results)[:, 0],clf)
    acc_results.sort(key=takeFirst, reverse=True)
    Bestresults=acc_results[:10]
    print(Bestresults)
    acc_results.sort(key=takeFirst, reverse=False)
    worsttrainers=acc_results[:10]
    print(worsttrainers)
    print('next')

    Supertrainers=[]
    for image_i_tuple in Bestresults: #looking at the best MNIST/1 data sets
        for j in range(10):
            for size_DS in range(5):
                Supertrainers += [SortedListofDigits[j][indexes_list[j][image_i_tuple[1] * 5 + size_DS]]]

    with open(filename_csv(clf), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([i for i in range(784)]) #header
        for img in Supertrainers[:100]:
            writer.writerow(img)

    acc_results = []
    indexes_list = [[] for j in range(10)]