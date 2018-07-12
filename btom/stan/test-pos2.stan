data {
  int P;
}
transformed data {
  int P2 = P*P;
  int PT = P * (P - 1) / 2;
}
parameters {
  simplex[P2] chol_unc;
}
transformed parameters {
  matrix[P, P] L_real;
  matrix[P, P] L_imag;
  matrix[P, P] rho_real;
  matrix[P, P] rho_imag;

  {
    int k = P;
    for (i in 1:P) {
      L_real[i,i] = sqrt(chol_unc[i]);
      L_imag[i,i] = 0;
      for (j in 1:i-1) {
        k = k + 1;
        L_real[i,j] = sqrt(chol_unc[k]);
        L_imag[i,j] = sqrt(chol_unc[k + PT]);
        L_real[j,i] = 0;
        L_imag[j,i] = 0;
      }
    }
  }

  rho_real = multiply_lower_tri_self_transpose(L_real) + multiply_lower_tri_self_transpose(L_imag);
  rho_imag = L_imag * L_real';
  rho_imag -= rho_imag';
}
model {

  target += 0;
}
