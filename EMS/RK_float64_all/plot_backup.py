import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  sys
sys.path.insert(0,'.')
import seaborn as sns
sns.set_style("darkgrid")

s1 = pd.read_csv('s1.txt', sep='\t', header=None)
s2 = pd.read_csv('s2.txt', sep='\t', header=None)
s3 = pd.read_csv('s3.txt', sep='\t', header=None)
s4 = pd.read_csv('s4.txt', sep='\t', header=None)
b1 = pd.read_csv('b1.txt', sep='\t', header=None)
b2 = pd.read_csv('b2.txt', sep='\t', header=None)
b3 = pd.read_csv('b3.txt', sep='\t', header=None)
b4 = pd.read_csv('b4.txt', sep='\t', header=None)

t = np.linspace(0, 300, 300000)

plt.plot(t, s1[0], label = '+ 10^(-5)')
plt.plot(t, s2[0], label = '+ 10^(-7)')
plt.plot(t, s3[0], label = '+ 10^(-9)')
plt.plot(t, s4[0], label = '+ 10^(-11)')

plt.plot(t, b1[0], label = '- 10^(-5)')
plt.plot(t, b2[0], label = '- 10^(-7)')
plt.plot(t, b3[0], label = '- 10^(-9)')
plt.plot(t, b4[0], label = '- 10^(-11)')


plt.legend()
plt.show()

plt.plot(t, s1[1], label = '+ 10^(-5)')
plt.plot(t, s2[1], label = '+ 10^(-7)')
plt.plot(t, s3[1], label = '+ 10^(-9)')
plt.plot(t, s4[1], label = '+ 10^(-11)')

plt.plot(t, b1[1], label = '- 10^(-5)')
plt.plot(t, b2[1], label = '- 10^(-7)')
plt.plot(t, b3[1], label = '- 10^(-9)')
plt.plot(t, b4[1], label = '- 10^(-11)')

plt.legend()
plt.show()