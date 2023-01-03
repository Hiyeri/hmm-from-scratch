from hmm_from_scratch.model import ProbabilityVector, ProbabilityMatrix

a1 = ProbabilityVector({'rain': 0.7, 'sun': 0.3})
a2 = ProbabilityVector({'rain': 0.6, 'sun': 0.4})
A = ProbabilityMatrix({'hot': a1, 'cold': a2})

print(A)
print(A.df)

b1 = ProbabilityVector({'0S': 0.1, '1M': 0.4, '2L': 0.5})
b2 = ProbabilityVector({'0S': 0.7, '1M': 0.2, '2L': 0.1})
B = ProbabilityMatrix({'0H': b1, '1C': b2})

print(B)
print(B.df)

P = ProbabilityMatrix.initialize(list('abcd'), list('xyz'))
print('Dot product:', a1 @ A)
print('Initialization:', P)
print(P.df)