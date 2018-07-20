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
	int<lower=0> m;                      // Number of measurement types
	matrix[D,D] M_real[m];               // Real part of measurement ops
	matrix[D,D] M_imag[m];               // Imag part of measurement ops

	matrix[D,K] X_real_est;
	matrix[D,K] X_imag_est;
	matrix[D,K] X_real_std;
	matrix[D,K] X_imag_std;

	// Binomial specific data model
	int<lower=0> n[m];                   // Total measurements per type
	int<lower=0> k[m];                   // Data
}
transformed data {
	matrix[m,D*D] Mvec_real;
	matrix[m,D*D] Mvec_imag;
	vector[D*K] X_real_err_mean = -to_vector(X_real_est) ./ to_vector(X_real_std);
	vector[D*K] X_real_err_std = inv(to_vector(X_real_std));
	vector[D*K] X_imag_err_mean = -to_vector(X_imag_est) ./ to_vector(X_imag_std);
	vector[D*K] X_imag_err_std = inv(to_vector(X_imag_std));

	for (idx_m in 1:m) {
		Mvec_real[idx_m] = to_row_vector(M_real[idx_m]);
		Mvec_imag[idx_m] = to_row_vector(M_imag[idx_m]);
	}
}
parameters {
	// real and imaginary parts of Ginibre relative to the given ests and stds
	matrix[D,K] X_real_err;
	matrix[D,K] X_imag_err;
}
transformed parameters {
	// real and imaginary parts of density matrix
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;

	{
		matrix[D,K] X_real = X_real_est + X_real_std .* X_real_err;
		matrix[D,K] X_imag = X_imag_est + X_imag_std .* X_imag_err;
		rho_real = X_real * X_real' + X_imag * X_imag';
		rho_imag = X_real * X_imag';
	}
	rho_imag -= rho_imag';
	{
		real t = trace(rho_real);
		rho_real = rho_real / t;
		rho_imag = rho_imag / t;
	}
}
model {
	// Ginibre DxK prior
	to_vector(X_real_err) ~ normal(X_real_err_mean, X_real_err_std);
	to_vector(X_imag_err) ~ normal(X_imag_err_mean, X_imag_err_std);

	k ~ binomial(n, Mvec_real * to_vector(rho_real) + Mvec_imag * to_vector(rho_imag));
}
