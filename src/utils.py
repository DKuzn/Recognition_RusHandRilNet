def get_labels():
    labels_open = open('../scripts/labels', 'r')
    labels_open = labels_open.read().split('\n')
    labels_open = [i.split(' ') for i in labels_open]
    labels = []
    for i in labels_open:
        if i[1] != 'invalid':
            labels.append(i[1])
    return sorted(labels)
