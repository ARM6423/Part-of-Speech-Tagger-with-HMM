from collections import defaultdict
from queue import PriorityQueue


def estimate_emission_params(train_path, k=1):
    token_tag_count = defaultdict(int)
    tag_count = defaultdict(int)
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token, tag = line.rsplit(' ', 1)
                token_tag_count[(token, tag)] += 1
                tag_count[tag] += 1
    emission_params = defaultdict(lambda: defaultdict(float))
    for (token, tag), count in token_tag_count.items():
        emission_params[token][tag] = count / (tag_count[tag] + k)
        emission_params['#UNK#'][tag] = k / (tag_count[tag] + k)
    return emission_params


def estimate_transition_params(train_path):
    transition_count = defaultdict(int)
    tag_count = defaultdict(int)
    START = 'START'
    STOP = 'STOP'
    with open(train_path, 'r', encoding='utf-8') as file:
        prev_tag = START
        for line in file:
            line = line.strip()
            if line:
                _, tag = line.rsplit(' ', 1)
                transition_count[(prev_tag, tag)] += 1
                tag_count[prev_tag] += 1
                prev_tag = tag
            else:
                transition_count[(prev_tag, STOP)] += 1
                tag_count[prev_tag] += 1
                prev_tag = START
    tag_count[STOP] = 0
    transition_params = defaultdict(lambda: defaultdict(float))
    for (tag1, tag2), count in transition_count.items():
        transition_params[tag1][tag2] = count / tag_count[tag1]
    return transition_params


def k_viterbi(emission_parameters, transition_parameters, sentence, k):
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
                    new_score = score * \
                        transition_parameters[u][v] * \
                        emission_parameters[word][v]
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


train_path = "Data/ES/train"
dev_in_path = "Data/ES/dev.in"
dev_out_path = "Data/ES/dev.out"
dev_predicted_path_2nd = "Data/ES/dev.p3.2nd.out"
dev_predicted_path_8th = "Data/ES/dev.p3.8th.out"

emission_params = estimate_emission_params(train_path)
transition_params = estimate_transition_params(train_path)

with open(dev_in_path, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path_2nd, 'w', encoding='utf-8') as file_write_2nd:
        with open(dev_predicted_path_8th, 'w', encoding='utf-8') as file_write_8th:
            sentences = []
            for line in file:
                word = line.strip()
                if word:
                    sentences.append(word)
                else:
                    predicted_tags_2nd = k_viterbi(
                        emission_params, transition_params, ' '.join(sentences), 2)
                    predicted_tags_8th = k_viterbi(
                        emission_params, transition_params, ' '.join(sentences), 8)

                    for word, tag in zip(sentences, predicted_tags_2nd):
                        file_write_2nd.write(f"{word} {tag}\n")
                    file_write_2nd.write("\n")

                    for word, tag in zip(sentences, predicted_tags_8th):
                        file_write_8th.write(f"{word} {tag}\n")
                    file_write_8th.write("\n")

                    sentences = []

# Load the actual tags from ES/dev.out
actual_tags = []
with open(dev_out_path, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        actual_tags.append(tag)

# Load the predicted tags from ES/dev.p3.2nd.out
predicted_tags_2nd = []
with open(dev_predicted_path_2nd, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        predicted_tags_2nd.append(tag)

# Load the predicted tags from ES/dev.p3.8th.out
predicted_tags_8th = []
with open(dev_predicted_path_8th, 'r', encoding='utf-8') as file:
    for line in file:
        tag = line.strip().rsplit(' ', 1)
        predicted_tags_8th.append(tag)

# Calculate metrics for ES, k=2
precision_2nd, recall_2nd, f_score_2nd = calculate_metrics(
    predicted_tags_2nd, actual_tags)
print("ES, k=2 - Precision:", precision_2nd)
print("ES, k=2 - Recall:", recall_2nd)
print("ES, k=2 - F-score:", f_score_2nd)

# Calculate metrics for ES, k=8
precision_8th, recall_8th, f_score_8th = calculate_metrics(
    predicted_tags_8th, actual_tags)
print("ES, k=8 - Precision:", precision_8th)
print("ES, k=8 - Recall:", recall_8th)
print("ES, k=8 - F-score:", f_score_8th)
