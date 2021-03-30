# Word Sense Disambiguation Scorer
# By Yousef Haggy, Intro to NLP, Spring 2021

# Takes in predicted word sense for ambigious words as well as a "key" text and measures accuracy
# Also provides a confusion matrix of predicted vs actual senses

# Algorithm:
# Read predicted senses and key senses
# For each instance
#    Store actual sense vs predicted sense in a confusion matrix dictionary
#    Update accuracy
# Output confusion matrix and accuracy

# Example Input/Output
# python scorer.py my-line-answers.txt line-key.txt 
# First arg: Sense labelled test text file name (.txt)
# Second arg: Sense labelled key text file name (.txt)

# Output:
# Accuracy: 0.8888888888888888
#       phone         product 
# phone    62       10      
# product  4        50  
import sys
import re

args = sys.argv

if len(args) < 3:
    print("Not enough args")
    quit()
test_txt = args[1]
key_txt = args[2]

test_answers = []
with open(test_txt,"r") as f:
    test_answers=f.read().split("\n")

answers = []
with open(key_txt,"r") as f:
    answers=f.read().split("\n")
confusion_matrx = {}
total = 0
correct = 0
for index, answer in enumerate(test_answers):
    if answer == '':
        break
    key = re.search(r'senseid=\"(.*)\"', answers[index]).group(1)
    test_answer = re.search(r'senseid=\"(.*)\"', test_answers[index]).group(1)
    
    confusion_matrx[key] = confusion_matrx.get(key, {})
    confusion_matrx[key][test_answer] = confusion_matrx[key].get(test_answer, 0) + 1
    if test_answer == key:
        correct+=1
    total+=1

print("Accuracy:", correct/total)

print(("      {:<8}"*len(confusion_matrx.keys())).format(*confusion_matrx.keys()))
for key in confusion_matrx.keys():
    line ="{:<8}".format(key)
    for pred_key in confusion_matrx.keys():
        line+=" {:<8}".format(confusion_matrx[key][pred_key])
    print(line)