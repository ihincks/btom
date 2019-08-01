data {
	int<lower=1> D;											 // Hilbert space dimension
	int<lower=1> K;                      // Number of Ginibre subsystems

	// Measurement operators (M_real+iM_imag should be positive, but < I)
	int<lower=0> m;                      // Number of measurement types
	int<lower=0> p;                      // Number of measurement outcomes per type
	matrix[D,D] M_real[m,p];             // Real part of measurement ops
	matrix[D,D] M_imag[m,p];             // Imag part of measurement ops

	// Counts in each type
	int<lower=0> k[m,p];
}
transformed data {
	// vectorize meas ops up front for efficiency
	matrix[m*p,D*D] Mvec_real;
	matrix[m*p,D*D] Mvec_imag;

	for (idx_m in 1:m) {
		for (idx_p in 1:p) {
			int idx = (idx_m - 1) * p + idx_p;
			Mvec_real[idx] = to_row_vector(M_real[idx_m, idx_p]);
			Mvec_imag[idx] = to_row_vector(M_imag[idx_m, idx_p]);
		}
	}
}
parameters {
	// real and imaginary parts of Ginibre; stan has no complex support
	matrix[D,K] X_real;
	matrix[D,K] X_imag;
}
transformed parameters {
	// real and imaginary parts of density matrix
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;

	rho_real = X_real * X_real' + X_imag * X_imag';
	rho_imag = X_real * X_imag';
	rho_imag -= rho_imag';
	{
		real t = trace(rho_real);
		rho_real = rho_real / t;
		rho_imag = rho_imag / t;
	}
}
model {
	// Ginibre DxK prior
	to_vector(X_real) ~ normal(0,1);
	to_vector(X_imag) ~ normal(0,1);

	{
		// compute all of the projections at once
		vector[m*p] probs = Mvec_real * to_vector(rho_real) + Mvec_imag * to_vector(rho_imag);

		// now every chunk of p values is a prob dist for some meas operator
		for (idx_m in 1:m) {
			k[idx_m] ~ multinomial(segment(probs, (idx_m - 1) * p + 1, p));
		}
	}

}
