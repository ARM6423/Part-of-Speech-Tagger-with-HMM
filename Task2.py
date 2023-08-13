from collections import defaultdict, Counter
from queue import PriorityQueue


def estimate_emission_params(train_path, k=1):
    token_tag_count = Counter()
    tag_count = defaultdict(int)
    
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token, tag = line.rsplit(' ', 1)
                token_tag_count[(token, tag)] += 1
                tag_count[tag] += 1
    
    emission_params = defaultdict(lambda: defaultdict(float))
    unk_denominator = k + sum(tag_count.values())
    
    for (token, tag), count in token_tag_count.items():
        emission_params[token][tag] = count / (tag_count[tag] + k)
        emission_params['#UNK#'][tag] = k / unk_denominator
    
    return emission_params

def estimate_transition_params(train_path):
    transition_count = defaultdict(Counter)
    tag_count = defaultdict(int)
    START = 'START'
    STOP = 'STOP'
    
    with open(train_path, 'r', encoding='utf-8') as file:
        prev_tag = START
        for line in file:
            line = line.strip()
            if line:
                _, tag = line.rsplit(' ', 1)
                transition_count[prev_tag][tag] += 1
                tag_count[prev_tag] += 1
                prev_tag = tag
            else:
                transition_count[prev_tag][STOP] += 1
                tag_count[prev_tag] += 1
                prev_tag = START
    
    transition_params = defaultdict(lambda: defaultdict(float))
    total_tags = sum(tag_count.values())
    
    for tag1, transitions in transition_count.items():
        for tag2, count in transitions.items():
            transition_params[tag1][tag2] = count / tag_count[tag1]
    
    transition_params[START][START] = 1.0  # Start tag always transitions to itself
    
    return transition_params

def add_start_stop_tags(sentence):
    return ["START"] + sentence + ["STOP"]

def viterbi(emission_parameters, transition_parameters, sentence, k=1):
    words = sentence.split()
    n = len(words)

    viterbi_matrix = defaultdict(lambda: defaultdict(PriorityQueue))

    viterbi_matrix[0]['START'].put((-1.0, ['START']))

    for t in range(1, n+1):
        word = words[t-1]
        if word not in emission_parameters:
            word = '#UNK#'
        for v in emission_parameters[word]:
            for u in viterbi_matrix[t-1]:
                for score, path in viterbi_matrix[t-1][u].queue:
                    transition_prob = transition_parameters[u][v]
                    emission_prob = emission_parameters[word][v]
                    new_score = score * transition_prob * emission_prob
                    new_path = path + [v]
                    viterbi_matrix[t][v].put((-new_score, new_path))
                    if viterbi_matrix[t][v].qsize() > k:
                        viterbi_matrix[t][v].get()  # Remove lowest score

    opt_paths = []
    for tag in viterbi_matrix[n]:
        for score, path in viterbi_matrix[n][tag].queue:
            opt_paths.append((-score, path[1:]))

    sorted_paths = sorted(opt_paths)
    if k <= len(sorted_paths):
        return sorted_paths[k-1][1]
    else:
        default = ['O'] * n
        return default

def calculate_metrics(predicted_tags, actual_tags):
    tp = sum(1 for pred, actual in zip(predicted_tags, actual_tags)
             if pred == actual and pred != 'O')
    fp = sum(1 for pred, actual in zip(predicted_tags, actual_tags)
             if pred != actual and pred != 'O')
    fn = sum(1 for pred, actual in zip(predicted_tags, actual_tags)
             if pred != actual and actual != 'O')

    precision = tp / (tp + fp) if tp + fp != 0 else 0.0
    recall = tp / (tp + fn) if tp + fn != 0 else 0.0
    f_score = 2 * precision * recall / \
        (precision + recall) if precision + recall != 0 else 0.0

    return precision, recall, f_score


train_path_es = "Data/ES/train"
dev_in_path_es = "Data/ES/dev.in"
dev_out_path_es = "Data/ES/dev.out"
dev_predicted_path_es = "Data/ES/dev.p2.out"

train_path_ru = "Data/RU/train"
dev_in_path_ru = "Data/RU/dev.in"
dev_out_path_ru = "Data/RU/dev.out"
dev_predicted_path_ru = "Data/RU/dev.p2.out"

# Estimation for ES
emission_params_es = estimate_emission_params(train_path_es)
transition_params_es = estimate_transition_params(train_path_es)

# Estimation for RU
emission_params_ru = estimate_emission_params(train_path_ru)
transition_params_ru = estimate_transition_params(train_path_ru)

# Dev set processing for ES
with open(dev_in_path_es, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path_es, 'w', encoding='utf-8') as file_write_es:
            sentences = []
            for line in file:
                word = line.strip()
                if word:
                    sentences.append(word)
                else:
                    sentences = add_start_stop_tags(sentences)
                    
                    predicted_tags_es = viterbi(
                        emission_params_es, transition_params_es, ' '.join(sentences))

                    for word, tag in zip(sentences[1:-1], predicted_tags_es[1:-1]):
                        file_write_es.write(f"{word} {tag}\n")
                    file_write_es.write("\n")

                    sentences = []

# Dev set processing for RU
with open(dev_in_path_ru, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path_ru, 'w', encoding='utf-8') as file_write_ru:
            sentences = []
            for line in file:
                word = line.strip()
                if word:
                    sentences.append(word)
                else:
                    sentences = add_start_stop_tags(sentences)
                    
                    predicted_tags_ru = viterbi(
                        emission_params_ru, transition_params_ru, ' '.join(sentences))
                    
                    for word, tag in zip(sentences[1:-1], predicted_tags_ru[1:-1]):
                        file_write_ru.write(f"{word} {tag}\n")
                    file_write_ru.write("\n")

                    sentences = []

# Load the actual tags from ES/dev.out
actual_tags_es = []
with open(dev_out_path_es, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        actual_tags_es.append(tag)

# Load the predicted tags from ES/dev.p2.out
predicted_tags_es = []
with open(dev_predicted_path_es, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        predicted_tags_es.append(tag)

# Separator
print("\n######## ES METRIC CALCULATIONS ########\n")

# Calculate metrics for ES, k=1
precision_es, recall_es, f_score_es = calculate_metrics(
    predicted_tags_es, actual_tags_es)
print("ES, - Precision:", precision_es)
print("ES, - Recall:", recall_es)
print("ES, - F-score:", f_score_es)

# Load the actual tags from RU/dev.out
actual_tags_ru = []
with open(dev_out_path_ru, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        actual_tags_ru.append(tag)

# Load the predicted tags from RU/dev.p2.out
predicted_tags_ru = []
with open(dev_predicted_path_ru, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        predicted_tags_ru.append(tag)
# Separator
print("\n######## RU METRIC CALCULATIONS ########\n")

# Calculate metrics for RU, k=1
precision_ru, recall_ru, f_score_ru = calculate_metrics(
    predicted_tags_ru, actual_tags_ru)
print("RU, - Precision:", precision_ru)
print("RU, - Recall:", recall_ru)
print("RU, - F-score:", f_score_ru)
