functions {
	real square_contract_real(real x, real y) {
		real ax = abs(x);
		real ay = abs(y);
		if (ax < 1e-12 || ay < 1e-12) {
			return 1.0;
		} else if (ax < ay) {
			return sqrt(1 + square(x/y));
		} else {
			return sqrt(1 + square(y/x));
		}
	}
	matrix square_contract(matrix x, matrix y) {
		matrix[rows(x), rows(x)] out;
		for (i in 1:rows(x)) {
			for (j in 1:rows(x)) {
				if (j > i) {
					out[i,j] = 0;
				} else {
					out[i,j] = square_contract_real(x[i,j], y[i,j]);
				}
			}
		}
		return out;
	}
}
data {
	int<lower=1> D;											 // Dimension
	real s;
}
transformed data {
	vector[D] alpha;
	for (i in 1:D) {
		alpha[i] = 1 / s;
	}
}
parameters {
	cholesky_factor_corr[D] X_real_sq;
	cholesky_factor_corr[D] X_imag_sq;
	simplex[D] X_diag;
}
transformed parameters {
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;
	{
		matrix[D,D] r = square_contract(X_real_sq, X_imag_sq);
		matrix[D,D] tmp = r .* diag_pre_multiply(sqrt(X_diag), X_real_sq);
		rho_imag = r .* diag_pre_multiply(sqrt(X_diag), X_imag_sq);
		rho_real = tmp * tmp' + rho_imag * rho_imag';
		rho_imag = rho_imag * tmp';
		rho_imag -= rho_imag';
	}
}
model {
	X_diag ~ dirichlet(alpha);
	X_real_sq ~ lkj_corr_cholesky(1);
	X_imag_sq ~ lkj_corr_cholesky(1);
}
