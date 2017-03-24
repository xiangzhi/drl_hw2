
import numpy as np
import matplotlib.pyplot as plt
import sys
import time
def main():
	
	graph = np.loadtxt(sys.argv[1])

	f, axarr = plt.subplots(2, sharex=True)
	num = np.linspace(0,graph[-1,0],1000)
	z = np.polyfit(graph[:,0],graph[:,1], 1)
	z = np.poly1d(z)
	axarr[0].plot(graph[:,0],graph[:,1])
	#axarr[0].plot([graph[0,0],graph[-1,0]],z)
	axarr[0].plot(num,z(num))
	axarr[0].set_xlabel("steps")
	axarr[0].set_ylabel("avg. reward")
	axarr[1].plot(graph[:,0],graph[:,2])
	axarr[1].set_ylabel("avg. length")
	axarr[1].set_xlabel("steps")
	plt.show()



if __name__ == '__main__':
	main()