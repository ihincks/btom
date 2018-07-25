data {
	int<lower=1> D;											 // Dimension
	int<lower=1> K;                      // Number of Ginibre subsystems
}
parameters {
	vector[(D-K)*K + K*(K+1)/2] X_vec;
}
transformed parameters {
	matrix[D,K] X;
	matrix[D,D] rho;

	{
		int cnt = 1;
		for (i in 1:D) {
			for (j in 1:K) {
				if (j > i) {
					X[i,j] = 0;
				} else {
					X[i,j] = X_vec[cnt];
					cnt += 1;
				}
			}
		}
	}

	rho = X * X';
	rho /= trace(rho);
}
model {
	for (i in 1:D) {
		for (j in 1:K) {
			if (j <= i) {
				X[i,j] ~ normal(0,square(max(K-i+1, 1)));
			}
		}
	}
}
