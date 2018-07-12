data {
  int P;
}
transformed data {
  vector[P] alpha = rep_vector(1.0/P, P);
}
parameters {
  simplex[P] sqr_scale;
  corr_matrix[P] Omega;
}
transformed parameters {
  matrix[P, P] Sigma = quad_form_diag(Omega, sqrt(sqr_scale));
}
model {
  
  target += dirichlet_lpdf(sqr_scale  | alpha);
  target += lkj_corr_lpdf(Omega | 3);
}
