###### H M M #######
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

data=pd.read_csv("/kaggle/input/train-csv/train.csv")
# transition prob from start state to any state

def start_trans_prob(data):
    initial_state_counts = {}
    start_state_prob={}
    
    for sentence in data:
        sentence=ast.literal_eval(sentence)
        x=sentence[0][1]
        if x not in initial_state_counts:
            initial_state_counts[x] = 1
        else:
            initial_state_counts[x] += 1
            
    n = sum(initial_state_counts.values())
    for tag,count in initial_state_counts.items():
        start_state_prob[tag] = count/n
    return start_state_prob

start_states_prob = start_trans_prob(data.iloc[:,1])

######################################################################################
# transition prob from state to state
def transition_prob(data):
    transition_counts={}
    unique=[]
    transition_probabilitie={}
    for s in data:
        s=ast.literal_eval(s)
        l=len(s)
        for i in range(l-1):
            if s[i][1] not in transition_counts:
                transition_counts[s[i][1]]={}
            if s[i+1][1] not in transition_counts[s[i][1]]:
                transition_counts[s[i][1]][s[i+1][1]]=1
            else:
                transition_counts[s[i][1]][s[i+1][1]]+=1
                
     # smoothing           
    for i,j in transition_counts.items():
        if i not in unique:
            unique.append(i)
            
    for i in unique:
        for k,m in transition_counts.items():
            if i not in transition_counts[k]:
                transition_counts[k][i]=0
                
    for m,p in transition_counts.items():
        for q in p:
            p[q]+=1
            
     # calculating transition probability           
    for current_tag,next_tag in transition_counts.items():
        if current_tag not in transition_probabilitie:
            transition_probabilitie[current_tag]={}
        n=sum(next_tag.values())

        for i,j in next_tag.items():
            transition_probabilitie[current_tag][i]=j/n
    return transition_probabilitie
 
transition_probabilities = transition_prob(data.iloc[:,1])

#######################################################################################
# emition prob
def emition_prob(data):
    emition_counts={}
    emition_probabilitie={}
    for s in data:
        s=ast.literal_eval(s)
        l=len(s)
        for i in range(l):
            if s[i][1] not in emition_counts:
                emition_counts[s[i][1]]={}
            if s[i][0] not in emition_counts[s[i][1]]:
                emition_counts[s[i][1]][s[i][0]]=1
            else:
                emition_counts[s[i][1]][s[i][0]]+=1
                
    for tag,words in emition_counts.items():
        if tag not in emition_probabilitie:
            emition_probabilitie[tag]={}
        n=sum(words.values())

        for i,j in words.items():
            emition_probabilitie[tag][i]=j/n
    return emition_probabilitie
 
emition_probabilities = emition_prob(data.iloc[:,1])

data1=pd.read_csv('/kaggle/input/assignment-1-nlp/test_small.csv')


#######################################################################################
def viterbi(data):
    output = []
    index = []
    tag=max(start_states_prob,key=start_states_prob.get)


    for w in range(len(data)):
        s = ast.literal_eval(data.iloc[w].iloc[2])
        index.append(data.iloc[w].iloc[1])
        out = []
        n = len(s)
        p_oov=""
        previous = ""
        previous_result = 0

        for i in range(n):
            if i == 0:
                key = []
                p1 = []
                for x, y in emition_probabilities.items():
                    if s[i] in y:
                        p1.append(y[s[i]])
                        key.append(x)

                state = []
                result = []

                if len(p1) == 0:
                    ###oov#######
                    if previous:
                        pass
                    else:
                        previous=tag
                    ss=max(transition_probabilities[previous],key=transition_probabilities[previous].get)
                    res=max(transition_probabilities[previous].values())
                    out.append((s[i], ss))
                    previous = ss
                    previous_result = res
                else:
                    for x, y in start_states_prob.items():
                        if x in key:
                            idx = key.index(x)
                            r = y * p1[idx]
                            result.append(r)
                            state.append(x)

                    j = result.index(max(result))
                    out.append((s[i], state[j]))
                    previous = state[j]
                    previous_result = result[j]

            else:
                key = []
                p1 = []

                for x, y in emition_probabilities.items():
                    if s[i] in y:
                        p1.append(y[s[i]])
                        key.append(x)

                state = []
                result = []

                if len(p1) == 0:
                    ###oov#######
                    if previous:
                        pass
                    else:
                        previous=tag
                    ss=max(transition_probabilities[previous],key=transition_probabilities[previous].get)
                    res=max(transition_probabilities[previous].values())
                    out.append((s[i], ss))
                    previous = ss
                    previous_result = res
                else:
                    if previous in transition_probabilities:
                        for x in transition_probabilities[previous]:
                            if x in key:
                                j = key.index(x)
                                y = transition_probabilities[previous][x]
                                r = y * p1[j] * previous_result
                                result.append(r)
                                state.append(x)

                        k = result.index(max(result))
                        out.append((s[i], state[k]))
                        previous = state[k]
                        previous_result = result[k]
                    else:
                        for l in range(len(p1)):
                            r = 1 * p1[l] * previous_result
                            result.append(r)
                            state.append(key[l])

                        m = result.index(max(result))
                        out.append((s[i], state[m]))
                        previous = state[m]
                        previous_result = result[m]

        output.append(out)

    output_str = [str(sublist) for sublist in output]
    output_result = pd.DataFrame({'id': index, 'tagged_sentence': output_str})

    return output_result

prediction_hmm = viterbi(data1) 
# print(prediction)
path_to_directory = '/kaggle/working/'
prediction_hmm.to_csv(path_to_directory +' sample_submission_hmm.csv', index = False)
