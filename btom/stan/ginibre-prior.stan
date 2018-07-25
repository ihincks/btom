data {
	int<lower=1> D;											 // Dimension
	int<lower=1> K;                      // Number of Ginibre subsystems
}
parameters {
	// real and imaginary parts of Ginibre; stan has no complex support
	matrix[D,K] X_real;
	matrix[D,K] X_imag;
}
model {
	to_vector(X_real) ~ normal(0,1);
	to_vector(X_imag) ~ normal(0,1);
}
generated quantities {
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
