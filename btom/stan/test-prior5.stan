data {
	int<lower=1> D;											 // Dimension
	real s;                              // Dirichlet parameter
	real t;
}
transformed data {
	vector[D] powers;
	for (i in 1:D) {
		powers[i] = 2 * (D - i) + 1;
	}
	powers[1] += t;
}
parameters {
	// unconstrained cholesky diagonal
	real X_diag_unc[D-1];
	// unconstrained cholesky off-diagonals, real and imaginary
	real X_offdiag_unc[D*D-D];
}
transformed parameters {
	matrix[D,D] rho_real;
	matrix[D,D] rho_imag;
	real x[D*D-1];
	real y[D*D];
	real targ = 0;

	{
		matrix[D,D] X_real;
		matrix[D,D] X_imag;
		real X_vec[D*D];
		real square_sum;
		int cnt = D+1;

		// constrain the diags and off-diags of the cholesky factor
		X_vec[2:D] = inv_logit(X_diag_unc);
		X_vec[D+1:D*D] = tanh(X_offdiag_unc);
		// respective jacobians
		targ += sum(log(X_vec[2:D])) + sum(log1m(X_vec[2:D]));
		targ += sum(log1m(square(X_vec[D+1:D*D])));

		x = X_vec[2:D*D];

		// perform 2-norm stick breaking and sum jacobian along the way
		square_sum = square(X_vec[2]);
		for (i in 3:D*D) {
			targ += 0.5 * log1m(square_sum);
			X_vec[i] *= sqrt(1 - square_sum);
			square_sum += square(X_vec[i]);
		}
		targ += 0.5 * log1m(square_sum);
		X_vec[1] = sqrt(1 - square_sum);

		y = X_vec;

		// populate the lower-diagonal cholesky factor
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
					X_imag[i,j] = X_vec[cnt+1];
					cnt += 2;
				}
			}
		}
		// jacobian of multiplying cholesky factors + Dirichlet on density mat svds
		targ += sum(to_vector(log(X_vec[1:D])) .* powers[1:D]);
		targ += (2*s-1)*sum(log(X_vec[1:D]));

		rho_real = multiply_lower_tri_self_transpose(X_real);
		rho_real += multiply_lower_tri_self_transpose(X_imag);
		rho_imag = X_real * X_imag';
		rho_imag -= rho_imag';
	}
}
model {
	target += targ;
}
