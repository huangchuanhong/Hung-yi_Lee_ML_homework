import matplotlib.pyplot as plt

with open('train_log') as f:
    lines = f.readlines()
lines = filter(lambda f: 'loss' in f, lines)
g_losses = []
d_losses = []
fdata_losses = []
rdata_rlabel_losses = []
rdata_flabel_losses = []
kl_losses = []

epoch = 0
xticks = [0]
labels_ = [0]
for i, line in enumerate(lines):
    splits = line.strip().split(', ')
    e = int(splits[0].split('epoch:')[-1])
    if e == epoch + 1:
        xticks.append(i)
        labels_.append(e)
        epoch += 1
    dis_loss = float(splits[2].split('d_loss=')[-1])
    rdata_rlabel_loss = float(splits[3].split('r_loss=')[-1])
    rdata_flabel_loss = float(splits[4].split('w_loss=')[-1])
    fdata_loss = float(splits[5].split('f_loss=')[-1])
    gen_loss = float(splits[6].split('g_loss=')[-1])
    kl_loss = float(splits[7].split('kl_loss=')[-1])

    g_losses.append(gen_loss)
    d_losses.append(dis_loss)
    fdata_losses.append(fdata_loss)
    rdata_rlabel_losses.append(rdata_rlabel_loss)
    rdata_flabel_losses.append(rdata_flabel_loss)
    kl_losses.append(kl_loss)
    # print(f'd_loss={dis_loss}')
    # print(f'r_loss={rdata_rlabel_loss}')
    # print(f'w_loss={rdata_flabel_loss}')
    # print(f'f_loss={fdata_loss}')
    # print(f'g_loss={gen_loss}')
    # print(f'kl_loss={kl_loss}')
    # exit()

 
plt.figure(figsize=(8, 32))
plt.xticks(xticks, labels_)
plt.subplot(6, 1, 1)
plt.title('gen_loss')
plt.xticks(xticks, labels_)
plt.xlabel('epoch')
plt.plot(g_losses)
plt.subplot(6,1,2)
plt.title('dis_loss')
plt.xticks(xticks, labels_)
plt.xlabel('epoch')
plt.plot(d_losses)
plt.subplot(6, 1,3)
plt.title('fdata_loss')
plt.xticks(xticks, labels_)
plt.xlabel('epoch')
plt.plot(fdata_losses)
plt.subplot(6, 1,4)
plt.title('rdata_rlabel_loss')
plt.xticks(xticks, labels_)
plt.xlabel('epoch')
plt.plot(rdata_rlabel_losses)
plt.subplot(6, 1,5)
plt.title('rdata_flabel_loss')
plt.xlabel('epoch')
plt.xticks(xticks, labels_)
plt.plot(rdata_flabel_losses)
plt.subplot(6,1,6)
plt.title('kl_loss')
plt.xlabel('epoch')
plt.xticks(xticks, labels_)
plt.plot(kl_losses)
plt.title('kl_loss')
plt.savefig('loss.jpg')
