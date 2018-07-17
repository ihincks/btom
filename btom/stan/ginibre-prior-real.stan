data {
	int<lower=1> D;											 // Dimension
	int<lower=1> K;                      // Number of Ginibre subsystems
}
parameters {
	matrix[D,K] X;
}
transformed parameters {
	matrix[D,D] rho;

	rho = X * X';
	rho /= trace(rho);
}
model {
	// Ginibre DxK prior
	for (idx_row in 1:D) {
		X[idx_row] ~ normal(0,1);
	}
}
 	
