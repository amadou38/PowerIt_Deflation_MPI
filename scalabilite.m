

clear; close all; clc;

% Scalabilite forte
% N = 10
x=[1 2 4];
BLAS1 = [5.9e-06 3e-06 1.5e-06];      % Temps de calcul BLAS1 par processus
BLAS2 = [5.7e-05 2.9e-05 1.5e-05];	% Temps de calcul BLAS2 par processus (maxiter = 100)
BLAS3 = [0.00014 0.00007 0.00004];	% Temps de calcul BLAS3 par processus (maxiter = 1000)

plot([1 2 4], BLAS1(1)./BLAS1, '*-', [1 2 4], BLAS2(1)./BLAS2, '*-',[1 2 4], BLAS3(1)./BLAS3,'*-', x, x, 'r--');
title('Scalabilite forte')
% axis([1 4 1 4])
legend('BLAS1', 'BLAS2', 'BLAS3')
xlabel('Proc'); ylabel('Acceleration');

figure,
y = [1 1 1];
plot([1 2 4], BLAS1(1)./(BLAS1.*x), '*-', [1 2 4], BLAS2(1)./(BLAS2.*x), '*-',[1 2 4], BLAS3(1)./(BLAS3.*x),'*-', x, y, 'r--');
title('Scalabilite forte')
% axis([1 4 1 4])
legend('BLAS1', 'BLAS2', 'BLAS3')
xlabel('Proc'); ylabel('Efficacité');