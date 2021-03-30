# Word Sense Disambiguation using Decision Lists Model
# By Yousef Haggy
# Intro to NLP, Spring 2021

# Overview: Given a training text that identifies the word 'line' and its sense in a multiple contexts 
# as well as instances of the word 'line' without the sense labeled called 'test set', output senses for each instance
# of 'line' in our test set

# Detailed overview/algorithm:
# 1. Take in our training set and parse the XML
# 2. For each instance of 'line'
#    a. For each of the following collocations, update the count of the labelled sense occuring given the collocation:
        # Word to the right of line, 
        # word to the left, 
        # word at +/- k boundary,
        #  words at +1 and +2 offset, 
        # word at -1 and -2 offset, 
        # word at -1 +1 at offset
# 3. Compute the log likelihood ratio for each collocation and predicted sense:
#   a. abs(log((P(phone|collocation) + alpha)/(P(phone|collocation)+alpha)))
# 4. Output log likliehoods/sense to specified output file
# 5. Load unlabelled test set, parse the XML
# 6. For each instance of the word 'line':
#   a. generate the same collocations mentioedn in step 2a
#   b. For each collocation, add the predicted senses and loglikelihood ratios identifeid from training set to a list
#   c. Pick the sense with the highest loglikelihood from that list as our 'best sense', assign label, and move on
#   d. If there are no collocations matching the collocation founds in training data, use most frequent sense as label sense
# 7. Output all labeled instances

# MFS Baseline accuracy (assign 'phone' to everything): 0.5714285714285714
# Accuracy of this program: 0.8888888888888888
# Confusion matrix:
#       phone         product 
# phone    62       10      
# product  4        50    

# As you can see, this program outperformed the most frequent sense baseline by over 30%

# Example input/usage
# python wsd.py line-train.txt line-test.txt my-model.txt 
# First arg: Sense annotated training data file name (.txt)
# Second arg: Unlabeled test data file name (.txt)
# Third arg: File name to output model too

#  STD Output (Predicted senses for all instances in training data):
# <answer instance="line-n.w8_059:8174:" senseid="phone"/>
# <answer instance="line-n.w7_098:12684:" senseid="phone"/>
# <answer instance="line-n.w8_106:13309:" senseid="phone"/>
# <answer instance="line-n.w9_40:10187:" senseid="phone"/>
# <answer instance="line-n.w9_16:217:" senseid="product"/>
# <answer instance="line-n.w8_119:16927:" senseid="product"/>
# <answer instance="line-n.w8_008:13756:" senseid="phone"/>
# <answer instance="line-n.w8_041:15186:" senseid="phone"/>
# <answer instance="line-n.art7} aphb 05601797:" senseid="phone"/>
# <answer instance="line-n.w8_119:2964:" senseid="product"/>
# <answer instance="line-n.w7_040:13652:" senseid="phone"/>
# <answer instance="line-n.w7_122:2194:" senseid="phone"/>
# ...

#  my-model.txt
#  Collocation: line growth, Log-Likelihood Ratio: 1.041392685158225, Predicts Sense: phone, Frequencies: {'phone': 3} 
#  Collocation: access line, Log-Likelihood Ratio: 1.041392685158225, Predicts Sense: phone, Frequencies: {'phone': 15} 
#  Collocation: basic, Log-Likelihood Ratio: 1.041392685158225, Predicts Sense: phone, Frequencies: {'phone': 1} 
#  Collocation: 4.5% access line, Log-Likelihood Ratio: 1.041392685158225, Predicts Sense: phone, Frequencies: {'phone': 1} 
#  Collocation: line growth in, Log-Likelihood Ratio: 1.041392685158225, Predicts Sense: phone, Frequencies: {'phone': 1} 
# ...
import sys
import xml.etree.ElementTree as ET
from lxml import etree
import re
import math

# According to Yarowski's paper, for semantic based ambiguity k between 20-50 is optimal
k = 30
args = sys.argv

if len(args) < 4:
    print("Not enough args")
    quit()
training_txt = args[1]
test_txt = args[2]
output_model_name = args[3]

collocation_distributions = {}
collocation_counts = {}
sense_counts = {}
training_string = ""
with open(training_txt, "r") as f:
    training_string = f.read()

# Do some cleanup to training string so that we can use XML parser
training_string = re.sub('(<@>|</p>|<p>|<s>|</s>)','', training_string)
training_string = re.sub('&','&amp;',training_string)
training_string = re.sub(' +',' ', training_string)
training_tree = ET.fromstring(training_string, parser=ET.XMLParser(encoding="UTF-8"))

root = training_tree.find('lexelt')
instances = root.findall('instance')

# Utility method for updated collocation distributions
def update_collocation_distribution(collocation, sense):
    collocation_distributions[collocation] =  collocation_distributions.get(collocation, {})
    collocation_distributions[collocation][sense] = collocation_distributions[collocation].get(sense,0) + 1
    collocation_counts[collocation] = collocation_counts.get(collocation,0) + 1

for instance in instances:
    answer = instance.find('answer')
    sense = answer.attrib["senseid"]
    context = instance.find('context')
    head = context.find('head')

    words = ET.tostring(context, encoding="unicode").replace("\n","").split(" ")
    words = [word for word in words if word not in ["","<context>","</context>"]]
    
    index_of_head = words.index(f'<head>{head.text}</head>')
    # Update collocation distributions for each collaction
    # 1. Word immediately to the right
    if index_of_head + 1 < len(words):
        word = words[index_of_head+1]
        collocation = f'line {word}'
        update_collocation_distribution(collocation, sense)
    # 2. Word immediately to the left
    if index_of_head -1 >=0:
        word = words[index_of_head-1]
        collocation = f'{word} line'
        update_collocation_distribution(collocation, sense)
    # 3a. Word in +k window
    if index_of_head + k < len(words):
        word = words[index_of_head+k]
        collocation = f'{word}'
        update_collocation_distribution(collocation, sense)
    # 3b. Word in -k window
    if index_of_head - k >= 0:
        word = words[index_of_head-k]
        collocation = f'{word}'
        update_collocation_distribution(collocation, sense)
    # 4. Words -2 and -1 offset
    if index_of_head -2 >=0:
        word1 = words[index_of_head-2]
        word2 = words[index_of_head-1]
        collocation = f'{word1} {word2} line'
        update_collocation_distribution(collocation, sense)
    # 5. Words +1 and +2 offset
    if index_of_head +2 < len(words):
        word1 = words[index_of_head+1]
        word2 = words[index_of_head+2]
        collocation = f'line {word1} {word2}'
        update_collocation_distribution(collocation, sense)
    # 6. Words +1 and -1 offset
    if index_of_head + 1 < len(words) and index_of_head -1 >=0:
        word1 = words[index_of_head-1]
        word2 = words[index_of_head+1]
        collocation = f'{word1} line {word2}'
        update_collocation_distribution(collocation, sense)

    sense_counts[sense] = sense_counts.get(sense,0) + 1
# Compute log likelihood for all collocations
log_likelihoods = {}
for collocation, value in collocation_distributions.items():
    # Detailed in Yarowski's paper, a small constant to handle the case where denominator is 0
    alpha = .1
    prob_phone = value.get("phone",0) / collocation_counts[collocation]
    prob_product = value.get("product",0) / collocation_counts[collocation]
    log_likelihood = abs(math.log( ((prob_phone + alpha) / (prob_product + alpha)) ,10))
    sense = "phone" if prob_phone > prob_product else "product"
    log_likelihoods[collocation]={"log_likelihood": log_likelihood, "sense":sense, "value":value}

# Save log likelihoods and features in descending order
with open(output_model_name,"w") as f:
    model = ""
    for collocation, data in log_likelihoods.items():
        model += f"Collocation: {collocation}, Log-Likelihood Ratio: {data['log_likelihood']}, Predicts Sense: {data['sense']}, Frequencies: {data['value']} \n\n "
    f.write(model)

# Now let's apply our model to the input test file
test_string = ""
with open(test_txt, "r") as f:
    test_string = f.read()

# Do some cleanup to training string so that we can use XML parser
test_string = re.sub('(<@>|</p>|<p>|<s>|</s>)','', test_string)
test_string = re.sub('&','&amp;',test_string)
test_string = re.sub(' +',' ', test_string)
test_tree = ET.fromstring(test_string, parser=ET.XMLParser(encoding="UTF-8"))

root = test_tree.find('lexelt')
instances = root.findall('instance')
result = ""

for instance in instances:
    id = instance.attrib["id"]
    context = instance.find('context')
    head = context.find('head')

    words = ET.tostring(context, encoding="unicode").replace("\n","").split(" ")
    words = [re.sub(r'[,|.|:|?]','',word) for word in words if word not in ["","<context>","</context>"]]
    
    index_of_head = words.index(f'<head>{head.text}</head>')
    # Get high log-likelihood sense for each evidence/collocation
    possible_sense = []
    # 1. Word immediately to the right
    word = words[index_of_head+1]
    collocation = f'line {word}'
    possible_sense.append(log_likelihoods.get(collocation,None))
    # 2. Word immediately to the left
    word = words[index_of_head-1]
    collocation = f'{word} line'
    possible_sense.append(log_likelihoods.get(collocation,None))
    # 3a. Word in +k window
    if index_of_head + k < len(words):
        word = words[index_of_head+k]
        collocation = f'{word}'
        possible_sense.append(log_likelihoods.get(collocation,None))

    # 3b. Word in -k window
    if index_of_head - k >= 0:
        word = words[index_of_head-k]
        collocation = f'{word}'
        possible_sense.append(log_likelihoods.get(collocation,None))    
    # 4. Words -2 and -1 offset
    if index_of_head -2 >=0:
        word1 = words[index_of_head-2]
        word2 = words[index_of_head-1]
        collocation = f'{word1} {word2} line'
        possible_sense.append(log_likelihoods.get(collocation,None))

    # 5. Words +1 and +2 offset
    if index_of_head +2 < len(words):
        word1 = words[index_of_head+1]
        word2 = words[index_of_head+2]
        collocation = f'line {word1} {word2}'
        possible_sense.append(log_likelihoods.get(collocation,None))

    # 6. Words +1 and -1 offset
    if index_of_head + 1 < len(words) and index_of_head -1 >=0:
        word1 = words[index_of_head-1]
        word2 = words[index_of_head+1]
        collocation = f'{word1} line {word2}'
        possible_sense.append(log_likelihoods.get(collocation,None))

    # Sort possible senses by highest log likelihood
    possible_sense = [x for x in possible_sense if x is not None]
    baseline_sense = "phone" if sense_counts["phone"] > sense_counts["product"] else "product"
    best_sense = max(possible_sense, default={"sense":baseline_sense}, key=lambda x: x["log_likelihood"])["sense"]

    answer = f"<answer instance=\"{id}\" senseid=\"{best_sense}\"/>\n"
    result+=answer
print(result) 