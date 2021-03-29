import sys
import xml.etree.ElementTree as ET
from lxml import etree
import re
import math

k = 10
args = sys.argv

if len(args) < 4:
    print("Not enough args")
    quit()
training_txt = args[1]
test_txt = args[2]
output_model_name = args[3]

collocation_distributions = {}
collocation_counts = {}
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
    # TODO, check bounds
    # 1. Word immediately to the right
    word = words[index_of_head+1]
    collocation = f'line {word}'
    update_collocation_distribution(collocation, sense)
    # 2. Word immediately to the left
    word = words[index_of_head-1]
    collocation = f'{word} line'
    update_collocation_distribution(collocation, sense)
    # 3a. Word in +k window
    word = words[min(index_of_head+k,len(words)-1)]
    collocation = f'{word}'
    update_collocation_distribution(collocation, sense)
    # 3b. Word in -k window
    word = words[max(index_of_head-k,0)]
    collocation = f'{word}'
    update_collocation_distribution(collocation, sense)
    # 4. Words -2 and -1 offset
    word1 = words[max(index_of_head-2,0)]
    word2 = words[max(index_of_head-1,0)]
    collocation = f'{word1} {word2} line'
    update_collocation_distribution(collocation, sense)
    # 5. Words +1 and +2 offset
    word1 = words[min(index_of_head+1,len(words)-1)]
    word2 = words[min(index_of_head+2,len(words)-1)]
    collocation = f'line {word1} {word2}'
    update_collocation_distribution(collocation, sense)
    # 6. Words +1 and -1 offset
    word1 = words[max(index_of_head-1, 0)]
    word2 = words[min(index_of_head+1,len(words)-1)]
    collocation = f'{word1} line {word2}'
    update_collocation_distribution(collocation, sense)

# Compute log likelihood for all collocations
log_likelihoods = []
for collocation, value in collocation_distributions.items():
    print(value)
    # Detailed in Yarowski's paper, a small constant to handle the case where denominator is 0
    alpha = .1
    prob_phone = value.get("phone",0) / collocation_counts[collocation]
    prob_product = value.get("product",0) / collocation_counts[collocation]
    log_likelihood = abs(math.log( ((prob_phone + alpha) / (prob_product + alpha)) ,10))
    print(collocation, log_likelihood)
    log_likelihoods.append(log_likelihood)
print(sorted(log_likelihoods))
    
    # Abs(Log(P r(AccentP attern1|Collocationi)P r(AccentP attern2|Collocationi)))

# 