from umatrix import *
eye = lambda order: umatrix(*[[int(i==j) for j in range(order)] for i in range(order)])
fill = lambda x, order, num_cols=None: umatrix(*[[x]*(num_cols if num_cols is not None else order)]*order)
zeros = lambda order, num_cols=None: fill(0, order, num_cols)
ones = lambda order, num_cols=None: fill(1, order, num_cols)

_round = lambda v, p=0: round(v, p) if not isinstance(v, complex) else round(v.real, p)+round(v.imag, p)*1j
typecheck = lambda val: isinstance(val, int) or isinstance(val, float) or isinstance(val, complex)

class umatrix(matrix):
	def __init__(self, *content, are_rows=True):
		super().__init__(*content, are_rows=True)

	def __setitem__(self, *args):
		sub = args[1]
		if isinstance(args[0], tuple):
			norm_slice = lambda s, rows: (s.start if s.start is not None else 0, s.stop if s.stop is not None else (len(self.rows) if rows else len(self.rows[0])), s.step if s.step is not None else 1)
			a00, a01 = args[0]
			a00_slice, a01_slice = [isinstance(x, slice) for x in [a00, a01]]
			if a00_slice:
				a00_iter = norm_slice(a00, True)
			if a01_slice:
				a01_iter = norm_slice(a01, False)
			if a00_slice:
				if a01_slice:
					assert all([len(s) == (a01_iter[1]-a01_iter[0])//a01_iter[2] for s in sub])
					assert all([typecheck(x) for y in sub for x in y])
					for i in range(*a00_iter):
						for j in range(*a01_iter):
							self.rows[i][j] = sub[(i//a00_iter[2])-a00_iter[0]][(j//a01_iter[2])-a01_iter[0]]
				else:
					assert len(sub) == (a00_iter[1]-a00_iter[0]+1)//a00_iter[2]
					assert all([typecheck(x) for x in sub])
					for i in range(*a00_iter):
						self.rows[i][a01] = sub[(i-a00_iter[0])//a00_iter[2]]
			elif a01_slice:
				assert len(sub) == (a01_iter[1]-a01_iter[0]+1)//a01_iter[2]
				assert all([typecheck(x) for x in sub])
				for j in range(*a01_iter):
					self.rows[a00][j] = sub[(j-a01_iter[0])//a01_iter[2]]
		else:
			check = len(self.cols) - (args[0].start if isinstance(args[0], slice) else 0)
			assert len(sub) == check, "Replacement has length {}, should be {}".format(len(sub), check)
			assert all([typecheck(x) for y in sub for x in y] if isinstance(args[0], slice) else [typecheck(x) for x in sub])
			self.rows[args[0]] = sub

	def size(self):
		return len(self.rows) * len(self.rows[0])
