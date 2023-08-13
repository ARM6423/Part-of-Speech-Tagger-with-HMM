import numpy as np

def transition_helper(u,v,tags):
    count_u=0
    count_u_to_v=0
    for i in range(len(tags)):
        if tags[i]==u:
            count_u+=1
            if tags[i+1]==v:
                count_u_to_v+=1
    return count_u_to_v/count_u

def transition(tags):
    tags.insert(0,"START")
    tags.append("STOP")
    transition_parameters={}
    for i in range(len(tags)):
        if tags[i]=="STOP":
            break
        if (tags[i],tags[i+1]) not in transition_parameters:
            transition_parameters[(tags[i],tags[i+1])]=transition_helper(tags[i],tags[i+1],tags)
    del tags[0]
    del tags[-1]
    return transition_parameters

def emission_helper(x, y,tags, words=None, k=1):
    count_y=0
    if x=="#UNK#":
        for i in tags:
            if i==y:
                count_y+=1
        return k/(count_y+k)
    count_y_to_x=0
    for i in range(len(tags)):
        if tags[i] == y:
            count_y += 1
            if words[i] == x:
                count_y_to_x += 1
    return (count_y_to_x) / (count_y + k)

def emission(tags,train_words,test_words):
    emission_word_tag={}
    for x in np.unique(np.array(test_words)):
        y_val={}
        for y in np.unique(np.array(tags)):
            if x in train_words:
                ep = emission_helper(x, y,tags,train_words)
                y_val.update({y:ep})
            else:
                ep = emission_helper("#UNK#", y,tags)
                y_val.update({y:ep})
        emission_word_tag[x] = y_val
    return emission_word_tag

def viterbi(sequence, tags, transition_parameters, emission_parameters):
    n = len(sequence)
    num_tags = len(tags)

    LTR=np.zeros((num_tags,n))
    RTL=np.zeros((num_tags,n))


    first_word=sequence[0]
    for i in range(num_tags):
        if ("START", tags[i]) not in transition_parameters:
            LTR[i, 0] = 1e-10*emission_parameters[first_word][tags[i]]
        else:
            LTR[i, 0] = transition_parameters[("START", tags[i])] * emission_parameters[first_word][tags[i]]

    for w in range(1, n):
        for i in range(0,num_tags):
            max_prob = -1
            max_backpointer = -1
            for j in range(0,num_tags):
                if (tags[j],tags[i]) not in transition_parameters:
                    prob = LTR[j, w-1] * 1e-10 * emission_parameters[sequence[w]][tags[i]]
                else:
                    prob = LTR[j, w-1] * transition_parameters[(tags[j], tags[i])] * emission_parameters[sequence[w]][tags[i]]

                if prob > max_prob:
                    max_prob = prob
                    max_backpointer = j
            
            LTR[i, w] = max_prob
            RTL[i, w] = max_backpointer
    
    stop_max_prob = -1
    stop_max_backpointer = -1
    for i in range(num_tags):
        if (tags[i],"STOP") not in transition_parameters:
            prob = LTR[i, n-1] * 1e-10
        else:
            prob = LTR[i, n-1] * transition_parameters[(tags[i], "STOP")]

        if prob > stop_max_prob:
            stop_max_prob = prob
            stop_max_backpointer = i
    
    # Retrieve the best path using backpointers
    best_path = []
    for w in range(n-1,-1,-1):
        best_path.insert(0,tags[stop_max_backpointer])
        if w==0:
            break
        stop_max_backpointer=int(RTL[stop_max_backpointer,w])

    return best_path

def process_dataset(train_path, dev_in_path, dev_out_path, dev_predicted_path):
    train_file = open(train_path, "r", encoding="utf-8")
    train_words = []
    tags = []
    # ... (your previous train file reading code)
    for l in train_file:
        if l!="\n":
            lst=l.split()
            x=""
            for i in range(len(lst)-1):
                x+=lst[i]+" "
            x=x[0:-1]
            y=lst[-1]
            train_words.append(x)
            tags.append(y)

    test_file = open(dev_in_path, "r", encoding="utf-8")
    test_words = []
    # ... (your previous test file reading code)
    for line in test_file:
        if line.strip():  # Non-empty line
            sequence.append(line.strip())
        else:  # Empty line indicates end of sequence
            predicted_tags = viterbi(sequence, np.unique(tags), transition_parameters, emission_word_tag)
            for word, tag in zip(sequence, predicted_tags):
                pred_output.write(f"{word} {tag}\n")
            pred_output.write("\n")
            pred_output.flush()
            sequence = []

    emission_word_tag = emission(tags, train_words, test_words)
    transition_parameters = transition(tags)

    pred_output = open(dev_predicted_path, "w", encoding="utf-8")
    sequence = []

    for l in test_file:
        if l != "\n":
            sequence.append(l.strip())
        else:
            predicted_tags = viterbi(
                sequence, np.unique(tags), transition_parameters, emission_word_tag)
            for i in range(len(predicted_tags)):
                pred_output.write(sequence[i] + " " + predicted_tags[i] + "\n")
            pred_output.write("\n")
            sequence = []

    pred_output.close()

# Paths for both datasets
train_path_es = "Data/ES/train"
dev_in_path_es = "Data/ES/dev.in"
dev_out_path_es = "Data/ES/dev.out"
dev_predicted_path_es = "Data/ES/dev.p2.out"

train_path_ru = "Data/RU/train"
dev_in_path_ru = "Data/RU/dev.in"
dev_out_path_ru = "Data/RU/dev.out"
dev_predicted_path_ru = "Data/RU/dev.p2.out"

# Process the ES dataset
process_dataset(train_path_es, dev_in_path_es, dev_out_path_es, dev_predicted_path_es)

# Process the RU dataset
process_dataset(train_path_ru, dev_in_path_ru, dev_out_path_ru, dev_predicted_path_ru)