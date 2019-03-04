from random import shuffle

data = []
with open('data', 'r') as f:
    for i in range(150):
        data.append(f.readline())

shuffle(data)
with open('data', 'w') as f:
    for i in range(150):
        f.write(data[i])

