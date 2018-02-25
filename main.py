import sys, codecs
from entropy import EntropyCalc, Smoothing


if sys.argv[1] == "-a":

	text = sys.stdin.read()
	e = EntropyCalc(text, mess='none')
	print("base:\t{}".format(e.textEntropy()))

	for mess in ['char', 'word']:
		for prob in [10, 5, 1, 0.1, 0.01, 0.001]:
			e = EntropyCalc(text, mess=mess, prob=prob)
			print("{}:{}\t{}".format(mess, prob, e.textEntropy()))

elif sys.argv[1] == "-b":
	text = sys.stdin.read()
	s = Smoothing(text)
	# s.E(debug=True)
	print(s.crossEntropy())

else:
	print("fail")
