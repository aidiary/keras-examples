import os
import matplotlib.pyplot as plt

result_file1 = os.path.join('result_without_datagen', 'history.txt')
result_file2 = os.path.join('result_with_datagen', 'history.txt')

def load_results(filename):
    epoch_list = []
    val_loss_list = []
    val_acc_list = []

    with open(filename) as fp:
        fp.readline()  # skip title
        for line in fp:
            line = line.rstrip()
            cols = line.split('\t')

            assert len(cols) == 5

            epoch = int(cols[0])
            loss = float(cols[1])
            acc = float(cols[2])
            val_loss = float(cols[3])
            val_acc = float(cols[4])

            epoch_list.append(epoch)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)

    return epoch_list, val_loss_list, val_acc_list

epoch1, val_loss1, val_acc1 = load_results(result_file1)
epoch2, val_loss2, val_acc2 = load_results(result_file2)

plt.figure()
plt.plot(epoch1, val_loss1, 'b-', marker='.', label='without datagen')
plt.plot(epoch2, val_loss2, 'r-', marker='.', label='with datagen')
plt.grid()
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val_loss')
# plt.show()
plt.savefig('val_loss.png')

plt.figure()
plt.plot(epoch1, val_acc1, 'b-', marker='.', label='without datagen')
plt.plot(epoch2, val_acc2, 'r-', marker='.', label='with datagen')
plt.grid()
plt.legend(loc='lower right')
plt.xlabel('epoch')
plt.ylabel('val_acc')
# plt.show()
plt.savefig('val_acc.png')
