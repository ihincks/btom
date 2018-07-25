data {
	int<lower=1> D;											 // Dimension
	real s;                              // Dirichlet parameter
}
transformed data {
	vector[D] powers;
	for (i in 1:D) {
		powers[i] = 2 * (D - i) + 1 + (2*s - 1);
	}
}
parameters {
	// unconstrained cholesky diagonal
	real X_diag_unc[D-1];
	// unconstrained cholesky off-diagonals, real and imaginary
	real X_offdiag_unc[D*D-D];
}
transformed parameters {
	matrix[D,D] X_real;
	matrix[D,D] X_imag;
	real X_vec[D*D];
	real X_vec_sbl[D*D-1];
	real X_vec_norms[D*D-1];

	// constrain the diags and off-diags of the cholesky factor
	X_vec_sbl[1:D-1] = inv_logit(X_diag_unc);
	X_vec_sbl[D:D*D-1] = tanh(X_offdiag_unc);

	// perform 2-norm stick breaking
	{
		real square_sum = square(X_vec_sbl[1]);
		for (i in 3:D*D) {
			X_vec_norms[i-1] = sqrt(1 - square_sum);
			X_vec[i] = X_vec_sbl[i-1] * X_vec_norms[i-1];
			square_sum += square(X_vec[i]);
		}
		X_vec_norms[1] = sqrt(1 - square_sum);
		X_vec[1] = X_vec_norms[1];
		X_vec[2] = X_vec_sbl[1];
	}

	// populate the lower-diagonal cholesky factor
	{
		int cnt = D+1;

		for (i in 1:D) {
			for (j in 1:D) {
				if (j > i) {
					X_real[i,j] = 0;
					X_imag[i,j] = 0;
				} else if (j == i) {
					X_real[i,j] = X_vec[i];
					X_imag[i,j] = 0;
				} else {
					X_real[i,j] = X_vec[cnt];
					cnt += 1;
					X_imag[i,j] = X_vec[cnt];
					cnt += 1;
				}
			}
		}
	}
}
model {
	// jacobian of parameter constraints
	target += sum(log(X_vec_sbl[1:D-1])) + sum(log1m(X_vec_sbl[1:D-1]));
	target += sum(log1m(square(X_vec_sbl[D:D*D-1])));

	// jacobian of stick breaking
	target += sum(log(X_vec_norms));

	// jacobian of cholesky multiplication + Dirichlet on density singular values
	target += sum(to_vector(log(X_vec[1:D])) .* powers);
}
generated quantities {
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;

	rho_real = X_real * X_real' + X_imag * X_imag';
	rho_imag = X_real * X_imag';
	rho_imag -= rho_imag';
}
