data {
	int<lower=1> D;											 // Hilbert space dimension
	int<lower=1> K;                      // Number of Ginibre subsystems

	// Measurement operators (M_real+iM_imag should be positive, but < I)
	int<lower=0> m;                      // Number of measurement types
	matrix[D,D] M_real[m];               // Real part of measurement ops
	matrix[D,D] M_imag[m];               // Imag part of measurement ops

	// Binomial specific data model
	int<lower=0> n[m];                   // Total measurements per type
	int<lower=0> k[m];                   // Data
}
transformed data {
	matrix[m,D*D] Mvec_real;
	matrix[m,D*D] Mvec_imag;

	for (idx_m in 1:m) {
		Mvec_real[idx_m] = to_row_vector(M_real[idx_m]);
		Mvec_imag[idx_m] = to_row_vector(M_imag[idx_m]);

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

	k ~ binomial(n, Mvec_real * to_vector(rho_real) + Mvec_imag * to_vector(rho_imag));
}
