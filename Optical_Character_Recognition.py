# Assignment: Cpts 315 HW 3 - Optical Character Recognition
# Programmer: Rahul Singal (11471764)
# Date Created: 3/20/18
# Date Last Edited: 3/22/18
# Description: 
# Collaborators: Prof. Doppa in class pseduocode & slides

import re
from sklearn.linear_model import perceptron
from numpy import dot

# *********************** Reading In Files *************************
filename = open('ocr_train.txt', 'r')
Train_Data = filename.read().splitlines()
filename.close()

filename = open('ocr_test.txt', 'r')
Test_Data = filename.read().splitlines()
filename.close()

# *********************** Pre-Processing *************************
'''
Data Format: \2\t\im00....00\tm\t
there are 128 binary values after im which is the input label
there is a single character after the \t following the last binary value
'''

TrainInputValue = []
TrainOutputValue = []

for line in Train_Data:
    try:
        line = re.split('im', line)
        line = re.split('\t', line[1])
        TrainInputValue.append(line[0])
        TrainOutputValue.append(line[1])
    except IndexError:
        pass

for x in range (0, len(TrainInputValue)):
    temp = []
    temp.extend(TrainInputValue[x])
    temp = map(int, temp)
    TrainInputValue[x] = list(map(int, temp))

TestInputValue = []
TestOutputValue = []

for line in Test_Data:
    try:
        line = re.split('im', line)
        line = re.split('\t', line[1])
        TestInputValue.append(line[0])
        TestOutputValue.append(line[1])
    except IndexError:
        pass

for x in range (0, len(TestInputValue)):
    temp = []
    temp.extend(TestInputValue[x])
    temp = map(int, temp)
    TestInputValue[x] = list(map(int, temp))

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
            'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

"""
Online Multi-Class Classifier Learning Algorithm
Input: D = Training Examples, K = 26 (Num of classes), T = 20 (Iterations)
Output: w1,w2,w3,...,w26 (Weight vectors for each class) -> size 128 since 128 pixels

w = []
for index in range(0, 26):
    temp = []
    for x in range(0, 128)
        temp.append(0)
    w.append(temp)

for index in range(0, T):
    num_mistakes = 0
    accuracy = 0
    for y in range(0,len(TrainInputValue)):
        score = []
        Xt = TrainInputValue[y]
        Yt = TrainOutputValue[y]
        #Calculate weight of each class dot Xt, find the largest one and that will be Yht
        for x in range(0,26):    
            score.append(dot(w[x], Xt))
        #Find max score
        max_value = max(score)
        max_index = score.index(max_value)
        Yht = alphabet[max_index]
        if (Yht != Yt):
            #Update Yht and Yt
            YhT_index = alphabet.index(Yht)
            Yt_index = alphabet.index(Yt)
            w[Yt] = [x + y for x,y in zip(w[YT], Xt)]
            w[Yht] = [x - y for x,y in zip(w[Yht], Xt)]
            num_mistakes = num_mistakes + 1
        else:
            pass
    print("iteration " + str(index) + " had " + str(num_mistakes) + " mistakes")
    accuracy = (len(TrainInputValue) - num_mistakes) / len(TrainInputValue)    
    print("Iteration " + str(index) + "has an accuracy of " + str(accuracy) + "%")
"""

T = 20
w = []
for index in range(0, 26):
    temp = []
    for x in range(0, 128):
        temp.append(0)
    w.append(temp)

for index in range(0, T):
    num_mistakes = 0
    accuracy = 0
    for y in range(0,len(TrainInputValue)):
        score = []
        Xt = TrainInputValue[y]
        Yt = TrainOutputValue[y]
        #Calculate weight of each class dot Xt, find the largest one and that will be Yht
        for x in range(0,26):    
            score.append(dot(w[x], Xt))
        #Find max score
        max_value = max(score)
        max_index = score.index(max_value)
        Yht = alphabet[max_index]
        if (Yht != Yt):
            #Update Yht and Yt
            Yht_index = alphabet.index(Yht)
            Yt_index = alphabet.index(Yt)
            w[Yt_index] = [x + y for x,y in zip(w[Yt_index], Xt)]
            w[Yht_index] = [x - y for x,y in zip(w[Yht_index], Xt)]
            num_mistakes = num_mistakes + 1
        else:
            pass
    print("iteration " + str(index) + " had " + str(num_mistakes) + " mistakes")
    accuracy = (len(TrainInputValue) - num_mistakes) / len(TrainInputValue)    
    print("Iteration " + str(index) + " has an accuracy of " + str(accuracy) + "%")


print("**************TESTING DATA********************")
num_mistakes = 0
accuracy = 0
for x in range(0, len(TestInputValue)):
    score = []
    Xt = TestInputValue[x]
    Yt = TestOutputValue[x]
    for y in range(0,26):
        score.append(dot(w[y], Xt))
    max_value = max(score)
    max_index = score.index(max_value)
    Yht = alphabet[max_index]
    if (Yht != Yt):
        num_mistakes = num_mistakes + 1
    else:
        pass
            
accuracy = (len(TestInputValue) - num_mistakes) / len(TestInputValue)
print("Testing Mistakes: " + str(num_mistakes))
print("Testing Accuracy: " + str(accuracy))


print("**************REAL RESULTS********************")
practice = perceptron.Perceptron(max_iter=20)
practice.fit(TrainInputValue, TrainOutputValue)
print("Real Iteration-20 Accuracy: ") 
print (practice.score(TrainInputValue, TrainOutputValue)*100)
    
    