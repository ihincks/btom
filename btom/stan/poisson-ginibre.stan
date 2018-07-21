functions {
	vector get_flux(vector group_flux, int[] group_size) {
		int idx = 0;
		int n_groups = rows(group_flux);
		vector[sum(group_size)] flux;
		for (idx_group in 1:n_groups) {
			for (idx_subgroup in 1:group_size[idx_group]) {
				flux[idx+idx_subgroup] = group_flux[idx_group];
			}
			idx += group_size[idx_group];
		}
		return flux;
	}
}
data {
	int<lower=1> D;											 // Hilbert space dimension
	int<lower=1> K;                      // Number of Ginibre subsystems

	// Measurement operators (M_real+iM_imag should be positive, but < I)
	int<lower=0> m;                      // Number of measurement types
	matrix[D,D] M_real[m];               // Real part of measurement ops
	matrix[D,D] M_imag[m];               // Imag part of measurement ops

	int<lower=1> n_groups;
	int<lower=1> group_size[n_groups];
	real<lower=0> dark_flux_std;
	real<lower=0> dark_flux_est;

	// Poisson specific data model
	int<lower=0> counts[m];
}
transformed data {
	// vectorize measurement stuff
	matrix[m,D*D] Mvec_real;
	matrix[m,D*D] Mvec_imag;
	// dark flux hyperparams for gamma dist
	real dark_alpha;
	real dark_beta;
	int use_dark_flux_prior = dark_flux_std > 100 * machine_precision();

	for (idx_m in 1:m) {
		Mvec_real[idx_m] = to_row_vector(M_real[idx_m]);
		Mvec_imag[idx_m] = to_row_vector(M_imag[idx_m]);
	}

	if (use_dark_flux_prior) {
		// dark flux hyperparams for gamma dist
		dark_alpha = square(dark_flux_est / dark_flux_std);
		dark_beta = dark_flux_est / square(dark_flux_std);
	}
}
parameters {
	// real and imaginary parts of Ginibre; stan has no complex support
	matrix[D,K] X_real;
	matrix[D,K] X_imag;

	// total count flux
	vector<lower=0>[n_groups] group_flux;
	// dark count flux
	real<lower=0> dark_flux;
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

	// Use an improper prior on flux
	group_flux ~ normal(79000,500);

	{
		real dark_flux_value = dark_flux_est;
		if (use_dark_flux_prior) {
			dark_flux ~ gamma(dark_alpha, dark_beta);
			dark_flux_value = dark_flux;
		}

		counts ~ poisson(
			dark_flux_value +
			get_flux(group_flux, group_size) .* (Mvec_real * to_vector(rho_real) + Mvec_imag * to_vector(rho_imag))
		);
	}

}
