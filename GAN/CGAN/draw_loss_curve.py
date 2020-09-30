import matplotlib.pyplot as plt

with open('train_log') as f:
    lines = f.readlines()
lines = filter(lambda f: 'loss' in f and '[' in f and ']' in f, lines) 
g_losses = []
d_losses = []
fdata_losses = []
rdata_rlabel_losses = []
rdata_flabel_losses = []

epoch = 0
xticks = [0]
labels_ = [0]
for i, line in enumerate(lines):
    splits = line.strip().split(', ')
    e = int(splits[0].split()[0].split('epoch')[-1])
    if e == epoch + 1:
        xticks.append(i)
        labels_.append(e)
        epoch += 1
    gen_loss = float(splits[0].split('gen_loss=')[-1])
    dis_loss = float(splits[1].split('dis_loss=')[-1])
    fdata_loss = float(splits[2].split('fdata_loss=')[-1])
    rdata_rlabel_loss = float(splits[3].split('rdata_rlabel_loss=')[-1])
    rdata_flabel_loss = float(splits[4].split('rdata_flabel_loss=')[-1])
    g_losses.append(gen_loss)
    d_losses.append(dis_loss)
    fdata_losses.append(fdata_loss)
    rdata_rlabel_losses.append(rdata_rlabel_loss)
    rdata_flabel_losses.append(rdata_flabel_loss)

 
plt.figure(figsize=(8, 24))
plt.xticks(xticks, labels_)
plt.subplot(5, 1, 1)
plt.title('gen_loss')
plt.xticks(xticks, labels_)
plt.plot(g_losses)
plt.subplot(5,1,2)
plt.title('dis_loss')
plt.xticks(xticks, labels_)
plt.plot(d_losses)
plt.subplot(5, 1,3)
plt.title('fdata_loss')
plt.xticks(xticks, labels_)
plt.plot(fdata_losses)
plt.subplot(5, 1,4)
plt.title('rdata_rlabel_loss')
plt.xticks(xticks, labels_)
plt.plot(rdata_rlabel_losses)
plt.subplot(5, 1,5)
plt.title('rdata_flabel_loss')
plt.xticks(xticks, labels_)
plt.plot(rdata_flabel_losses)
plt.savefig('loss.jpg')
