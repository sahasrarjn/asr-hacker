import matplotlib.pyplot as plt

lstm_val_loss = [
    0.6997883737087249,
    0.6940132117271424,
    0.692995845079422,
    0.6930617749691009,
    0.6931677722930908
]

qlstm_val_loss = [
    0.6973038280010223,
    0.693778109550476,
    0.6933149290084839,
    0.6927791178226471,
    0.6934100031852722   
]

lstm_train_loss = [
    0.6953002727031707,
    0.6940183524290721,
    0.6940748528639475,
    0.6935462562243143,
    0.6936303385098775
]

qlstm_train_loss = [
    0.6957257692019144,
    0.6944602898756663,
    0.6940118050575257,
    0.6935296793778737,
    0.6938473983605703
]

lstm_val_acc = [
    50.12,
    49.88,
    49.8,
    50.28,
    49.559999999999995
]

qlstm_val_acc = [
    49.72,
    49.64,
    50.36000000000001,
    50.44,
    50.12
]

data1 = lstm_val_acc
data2 = qlstm_val_acc

epochs = [i for i in range(1,6)]

data1 = [l for l in data1]
data2 = [l for l in data2]

data1 = data1[:-1]
data2 = data2[:-1]
epochs = epochs[:-1]

plt.plot(epochs, data1, label='LSTM')
plt.plot(epochs, data2, label='QLSTM')
plt.xlabel('Epochs')
plt.ylabel('Validation Accuracy')
plt.xticks(epochs)
plt.legend()
plt.savefig('Val_acc.png')

