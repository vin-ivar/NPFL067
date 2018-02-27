import sys, codecs
import numpy as np
from entropy import EntropyCalc, Smoothing

if sys.argv[1] == "-a":

	text = sys.stdin.read()
	e = EntropyCalc(text, mess='none')
	print("base:\t{}".format(e.textEntropy()))
	for mess in ['char', 'word']:
		for prob in [0.001, 0.01, 0.1, 1, 5, 10]:
			vals = np.array([])	
			for i in range(10):
				e = EntropyCalc(text, mess=mess, prob=prob)
				entropy = e.textEntropy()
				vals = np.append(vals, entropy)

			print("{}:{}\t{}\t{}\t{}".format(mess, prob, min(vals), max(vals), vals.mean()))

elif sys.argv[1] == "-b":
	text = sys.stdin.read()
	s = Smoothing(text)
	# s.E(debug=True)
	print(s.crossEntropy())

else:
	print("fail")
