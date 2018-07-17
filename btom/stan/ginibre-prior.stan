data {
	int<lower=1> D;											 // Dimension
	int<lower=1> K;                      // Number of Ginibre subsystems
}
parameters {
	// real and imaginary parts of Ginibre; stan has no complex support
	matrix[D,K] X_real;
	matrix[D,K] X_imag;
}
transformed parameters {
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
	for (idx_row in 1:D) {
		X_real[idx_row] ~ normal(0,1);
		X_imag[idx_row] ~ normal(0,1);
	}
}
