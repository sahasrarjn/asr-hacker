from cProfile import label
import matplotlib.pyplot as plt


data_lstm = [
    0.00281,
    0.00227,
    0.00201,
    0.00187,
    0.00176,
    0.00170,
    0.00167,
    0.00165,
    0.00164,
    0.00163,
    0.00162,
    0.00161,
    0.00160,
    0.00159,
    0.00159,
    0.00158,
    0.00157,
    0.00156,
    0.00155,
    0.00155
]

data_qlstm = [
    0.00322,
    0.00262,
    0.00210,
    0.00180,
    0.00166,
    0.00163,
    0.00161,
    0.00160,
    0.00159,
    0.00158,
    0.00158,
    0.00157,
    0.00157,
    0.00156,
    0.00156,
    0.00155,
    0.00155,
    0.00155,
    0.00154,
    0.00154
]

epochs = [i*100 for i in range(21)]
epochs = epochs[1:]

data_lstm = [l*1000 for l in data_lstm]
data_qlstm = [l*1000 for l in data_qlstm]

plt.plot(epochs, data_lstm, label='LSTM')
plt.plot(epochs, data_qlstm, label='QLSTM')
plt.xlabel('Epochs')
plt.ylabel('Loss (x 1000)')
plt.legend()
plt.savefig('plot1.png')
