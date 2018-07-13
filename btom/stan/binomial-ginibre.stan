functions {
	real tr(matrix X, matrix Y) {
		real out = 0;
		for (i in 1:rows(X)) {
			for (j in 1:cols(X)) {
				out += X[i,j] * X[i,j] + Y[i,j] * Y[i,j];
			}
		}
		return out;
	}
}
data {
	int<lower=1> D;											 // Hilbert space dimension
	int<lower=1> K;                      // Number of Ginibre subsystems

	// Measurement operators (M_real+iM_imag should be positive, but < I)
	int<lower=1> m;                      // Number of measurement types
	matrix[D,D] M_real[m];               // Real part of measurement ops
	matrix[D,D] M_imag[m];               // Imag part of measurement ops

	// Binomial specific data model
	int<lower=1> n[m];                   // Total measurements per type
	int<lower=0> k[m];                   // Data
}
parameters {
	// real and imaginary parts of Ginibre; stan has no complex support
	matrix[D,K] X_real;
	matrix[D,K] X_imag;
}
model {
	// Ginibre DxK prior
	for (idx_row in 1:D) {
		X_real[idx_row] ~ normal(0,1);
		X_imag[idx_row] ~ normal(0,1);
	}

	{
		real t = tr(X_real, X_imag);
		for (idx_m in 1:m) {
			real p = (
				trace_quad_form(M_real[idx_m], X_real) +
				trace_quad_form(M_real[idx_m], X_imag) -
				2 * trace(X_imag' * M_imag[idx_m] * X_real)
			) / t;
			k[idx_m] ~ binomial(n[idx_m], p);
		}
	}
}
generated quantities {
	// real and imaginary parts of density matrix
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;

	rho_real = X_real * X_real' + X_imag * X_imag';
	rho_imag = X_real * X_imag';
	rho_imag -= rho_imag;
	{
		real t = trace(rho_real);
		rho_real = rho_real / t;
		rho_imag = rho_imag / t;
	}
}
