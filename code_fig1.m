% this file generates fig 1 in 
% 
% "Improved  convergence rate of $k$NN graph Laplacians" 
% by Yixuan Tan, Xiuyuan Cheng.



%%

clear all;

rng(1028);

%% choose probability density p

L_mfd = 1;

pdf0 = @(x,a,b) 1/L_mfd * (1 + sin(2*pi/L_mfd*a*x)*b);
cdf0 = @(x,a,b) 1/L_mfd * (x - L_mfd/(2*pi)*(cos(2*pi/L_mfd*a*x)-1)/a * b);

p_gradient0 = @(x,a,b) 2*pi / L_mfd^2 * cos(2*pi/L_mfd*a*x)*a*b;
p_laplacian0 = @(x,a,b) - 4*pi^2 / L_mfd^3 * sin(2*pi/L_mfd*a*x)*a^2*b;


a1 = 2; b1 = 1;
a2 = 3; b2 = 0.5;


pdf = @(x) 0.5 * pdf0(x,a1,b1) + 0.5 * pdf0(x,a2,b2);
cdf = @(x) 0.5 * cdf0(x,a1,b1) + 0.5 * cdf0(x,a2,b2);
p_gradient  = @(x) 0.5 * p_gradient0(x,a1,b1) + 0.5 * p_gradient0(x,a2,b2);
p_laplacian = @(x) 0.5 * p_laplacian0(x,a1,b1) + 0.5 * p_laplacian0(x,a2,b2);


%%

N_samp_ref = round(L_mfd*10^4);
t_samp_ref = linspace(0,1,N_samp_ref);
y_samp_ref = cdf(t_samp_ref);


%%





%%

% Gaussian kernel
sigma_g = 1;%0.4;
k_g_funh = @(r) exp(-r/(4*sigma_g^2)) / sqrt(4*pi*sigma_g^2) / sigma_g^2; % m_2[k] / 2 = 1
% disk kernel
r_disk = 1;
k_d_funh = @(r) ((0 <= r) & (r <= r_disk^2)) / (1/3 * r_disk^3); % m_2[h] / 2 = 1



omegaM         = 5;
map_to_RD_func = @(t) L_mfd  /(2*pi) * 1/(sqrt(5))*[...
    cos(2*pi/L_mfd*t), ...
    sin(2*pi/L_mfd*t), ...
    2/omegaM*cos( 2*pi/L_mfd*omegaM*t), ...
    2/omegaM*sin( 2*pi/L_mfd*omegaM*t)];

% map_to_RD_func = @(t) L_mfd  /(2*pi) * [...
%     cos(2*pi/L_mfd*t), ...
%     sin(2*pi/L_mfd*t)];

Q = @(x) (p_laplacian(x)./ pdf(x) + pi^2/5*(4*omegaM^2+1)) / 6;




%% kNN estimation plot


N = 2000;
alpha_d = 2;

x_rand_coord = sort(rand(N,1));
x_rand_coord = interp1(y_samp_ref, t_samp_ref, x_rand_coord);
x_rand_embed = map_to_RD_func(x_rand_coord);


%-----------------------------------------------------------------
k_knn1 = 32;

rk = (k_knn1 / (alpha_d*N)).^2;
rho_bar_ref1 = zeros(N,1);
for k = 1: N
    x_tmp = x_rand_coord(k);
    C = [rk * Q(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_ref1(k) = abs(roots_tmp(3));
end


[~, knn_d] = knnsearch(x_rand_embed, x_rand_embed, 'K', k_knn1);
rho_hat_ref1 = knn_d(:,k_knn1)/(k_knn1 / (alpha_d*N));

%-----------------------------------------------------------------
k_knn2 = 64;

rk = (k_knn2 / (alpha_d*N)).^2;
rho_bar_ref2 = zeros(N,1);
for k = 1: N
    x_tmp = x_rand_coord(k);
    C = [rk * Q(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_ref2(k) = abs(roots_tmp(3));
end


[~, knn_d] = knnsearch(x_rand_embed, x_rand_embed, 'K', k_knn2);
rho_hat_ref2 = knn_d(:,k_knn2)/(k_knn2 / (alpha_d*N));

%-----------------------------------------------------------------
figure(1), clf;

subplot(1,2,1)
hold on
plot(x_rand_coord, rho_hat_ref1, '.', 'DisplayName', "$\hat\rho$", 'MarkerSize', 10, 'LineWidth', 3, 'Color', "#0072BD");
plot(x_rand_coord, rho_bar_ref1, 'DisplayName', "$\bar\rho_{r_k}$ (with correction)", 'MarkerSize', 3, 'LineWidth', 3, 'Color', "#D95319");
plot(x_rand_coord, 1 ./ pdf(x_rand_coord), '--', 'DisplayName', "$\bar\rho$  $\;(= p^{-1/d})$", 'MarkerSize', 3, 'LineWidth', 3, 'Color', '#EDB120');
grid on
legend('Location', 'northwest', 'Interpreter', 'latex', 'FontSize', 26);
L = legend; L.AutoUpdate = 'off';
% xline(x_ref_coord(plt_ind), '--', 'LineWidth', 2, 'DisplayName', "$x$", 'Color', 'black');
% xline(0.75, '-.', 'LineWidth', 1, 'DisplayName', "$x$");
% xline(0.95, '-.', 'LineWidth', 1, 'DisplayName', "$x$");
set(gca, 'FontSize', 26);
xticks(0:0.2:1);
ylim([0, 3.7])
xlabel('Intrinsic coodinate', 'Interpreter', 'latex');
ylabel('Bandwidth functions', 'Interpreter', 'latex');
str_title = strcat('$k = ', num2str(k_knn1), '$');
title(str_title, 'Interpreter', 'latex', 'FontSize', 30);


subplot(1,2,2)
hold on
plot(x_rand_coord, rho_hat_ref2, '.', 'DisplayName', "$\hat\rho$", 'MarkerSize', 10, 'LineWidth', 3, 'Color', "#0072BD");
plot(x_rand_coord, rho_bar_ref2, 'DisplayName', "$\bar\rho_{r_k}$ (with correction)", 'MarkerSize', 3, 'LineWidth', 3, 'Color', "#D95319");
plot(x_rand_coord, 1 ./ pdf(x_rand_coord), '--', 'DisplayName', "$\bar\rho$  $\;(= p^{-1/d})$", 'MarkerSize', 3, 'LineWidth', 3, 'Color', '#EDB120');
grid on
set(gca, 'FontSize', 26);
xticks(0:0.2:1);
ylim([0, 3.7])
xlabel('Intrinsic coodinate', 'Interpreter', 'latex');
% ylabel('Bandwidth functions', 'Interpreter', 'latex');
str_title = strcat('$k = ', num2str(k_knn2), '$');
title(str_title, 'Interpreter', 'latex', 'FontSize', 30);


