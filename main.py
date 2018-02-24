import sys
from entropy import EntropyCalc, Smoothing

'''
text = sys.stdin.read()
for mess in ['char', 'word']:
	for prob in [10, 5, 1, 0.1, 0.01, 0.001]:
		e = EntropyCalc(text, mess=mess, prob=prob)
		print("{}:{}\t{}".format(mess, prob, e.textEntropy()))
'''
text = sys.stdin.read()
s = Smoothing(text)
# s.E(debug=True)
print(s.crossEntropy())
# print(s.unigram['this'])
