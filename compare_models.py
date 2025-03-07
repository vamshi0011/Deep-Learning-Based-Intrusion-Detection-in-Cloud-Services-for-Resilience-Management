import pandas as pd
import matplotlib.pyplot as plt

gan_results = pd.read_csv('gan_results_comparison.csv')
dbn_results = pd.read_csv('dbn_results_comparison.csv')

plt.plot(gan_results['split'], gan_results['accuracy'], label='GAN', marker='o')
plt.plot(dbn_results['split'], dbn_results['accuracy'], label='DBN', marker='o')
plt.xlabel('Dataset Split')
plt.ylabel('Accuracy')
plt.title('GAN vs DBN Accuracy Comparison')
plt.legend()
plt.grid(True)
plt.show()
