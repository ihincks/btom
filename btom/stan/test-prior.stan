data {
	int<lower=1> D;											 // Dimension
	int<lower=1> K;                      // Number of Ginibre subsystems
}
transformed data {
	int N = 2*K*D - K*K;
}
parameters {
	simplex X_diag[K];
	real X_offdiag_vec[N-K];
}
transformed parameters {
	matrix[D,K] X_real;
	matrix[D,K] X_imag;
	real X_vec[N-K];

	{
		int real_cnt = K+1;
		int imag_cnt = K * (2*D - K + 1)/2 + 1;

		X_vec[2:K] = inv_logit(X_vec_diag);
		X_vec[K+1:N] = tanh(X_vec_offdiag);

		{
			real square_sum = square(X_vec[2]);
			for (i in 3:N) {
				X_vec[i] = X_vec[i] * sqrt(1 - square_sum);
				square_sum += square(X_vec[i]);
			}
			X_vec[1] = sqrt(1 - square_sum);
		}

		for (i in 1:D) {
			for (j in 1:K) {
				if (j > i) {
					X_real[i,j] = 0;
					X_imag[i,j] = 0;
				} else if (j == i) {
					X_real[i,j] = X_vec[i];
					X_imag[i,j] = 0;
				} else {
					X_real[i,j] = X_vec[real_cnt];
					X_imag[i,j] = X_vec[imag_cnt];
					real_cnt += 1;
					imag_cnt += 1;
				}
			}
		}
	}
}
model {
	X_vec_diag ~ normal(0,1);
	X_vec_offdiag ~ normal(0,1);
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
