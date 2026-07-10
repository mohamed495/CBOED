class LinearizedOperator:
    """Linear operator matrix free

    - matvec(v)  : v -> G v
    - rmatvec(w) : w -> G^T w
    - T          : adjoint operator
    - shape      : (n_out, n_in)
    """

    def __init__(self, matvec, rmatvec, shape):
        self._mv = matvec
        self._rmv = rmatvec

        self.shape = shape

    def matvec(self, v):
        return self._mv(v)

    def rmatvec(self, w):
        return self._rmv(w)

    @property
    def T(self):
        n_out, n_in = self.shape
        return LinearizedOperator(self._rmv, self._mv, (n_in, n_out))
