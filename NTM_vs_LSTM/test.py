import matplotlib.pyplot as plt

with open('test.txt') as f:
    lines = f.readlines()

losses = []
accs = []
for line in lines:
    splits = line.strip().split()
    loss = float(splits[2].split('=')[1].split(',')[0])
    acc = float(splits[3].split('=')[1])
    losses.append(loss)
    accs.append(acc)

#plt.figure()
#x = list(range(len(losses)))
#plt.plot(x, losses)
#plt.savefig('loss.jpg')

plt.figure()
x = list(range(len(accs)))
plt.plot(x, accs)
plt.savefig('acc.jpg')
