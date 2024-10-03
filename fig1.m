clear all;
% close all;

%%

rng("shuffle");

%% choose probability density p

L_mfd = 1;

pdf0 = @(x,a,b) 1/L_mfd * (1 + sin(2*pi/L_mfd*a*x)*b);
cdf0 = @(x,a,b) 1/L_mfd * (x - L_mfd/(2*pi)*(cos(2*pi/L_mfd*a*x)-1)/a * b);

p_gradient0 = @(x,a,b) 2*pi / L_mfd^2 * cos(2*pi/L_mfd*a*x)*a*b;
p_laplacian0 = @(x,a,b) - 4*pi^2 / L_mfd^3 * sin(2*pi/L_mfd*a*x)*a^2*b;


% a1 = 2; b1 = 1;%0.75;%
% a2 = 5; b2 = 0.6;% 0;%

a1 = 2; b1 = 1;%0.75;%
a2 = 3; b2 = 0.5;% 0;%

% a1 = 2; b1 = 0;%0.75;%
% a2 = 9; b2 = 0;% 0;%


pdf = @(x) 0.5 * pdf0(x,a1,b1) + 0.5 * pdf0(x,a2,b2);
cdf = @(x) 0.5 * cdf0(x,a1,b1) + 0.5 * cdf0(x,a2,b2);
p_gradient  = @(x) 0.5 * p_gradient0(x,a1,b1) + 0.5 * p_gradient0(x,a2,b2);
p_laplacian = @(x) 0.5 * p_laplacian0(x,a1,b1) + 0.5 * p_laplacian0(x,a2,b2);


%%

N_samp_ref = round(L_mfd*10^4);
t_samp_ref = linspace(0,1,N_samp_ref);
y_samp_ref = cdf(t_samp_ref);


%%

plt_tmp = linspace(0,L_mfd,1000);
figure(11), clf;
plot(plt_tmp, pdf(plt_tmp), 'LineWidth', 2);
hold on
plot(plt_tmp, 1/L_mfd*ones(size(plt_tmp)), 'b--', 'LineWidth', 2);
set(gca, 'FontSize', 16);
xlabel('Intrinsic coordinate', 'Interpreter', 'latex', 'FontSize', 22);
ylabel('$p$', 'Interpreter', 'latex', 'FontSize', 22);
ylim([0, 2]);
title('Probability densities', 'Interpreter', 'latex', 'FontSize', 26);
grid on
legend_string = strings(2,1);
legend_string(1) = "non-uniform $p$";
legend_string(2) = "uniform $p$";
legend(legend_string, 'Location', 'best', 'Interpreter', 'latex', 'FontSize', 22)


%%



%           Graph Laplacian - pointwise convergence




%%

% Gaussian kernel
sigma_g = 1;%0.4;
k_g_funh = @(r) exp(-r/(4*sigma_g^2)) / sqrt(4*pi*sigma_g^2) / sigma_g^2; % m_2[k] / 2 = 1
% disk kernel
r_disk = 1;
k_d_funh = @(r) ((0 <= r) & (r <= r_disk^2)) / (1/3 * r_disk^3); % m_2[h] / 2 = 1


% f
f0 = @(x, f_shift, c, d) sin(2*pi/L_mfd*c*(x-f_shift)) * d;
f0_gradient = @(x, f_shift, c, d) 2*pi/L_mfd*c*cos(2*pi/L_mfd*c*(x-f_shift)) * d;
f0_laplacian = @(x, f_shift, c, d) -(2*pi/L_mfd)^2*c^2*sin(2*pi/L_mfd*c*(x-f_shift)) * d;

f_shift1 = 0.1;  c1 = 1; d1 = 1;
% f_shift1 = 0.55;  c1 = 1; d1 = 1;
f_shift2 = 5/11;   c2 = 5; d2 = 0;
f = @(x) 0.5 * f0(x, f_shift1, c1, d1) + 0.5 * f0(x, f_shift2, c2, d2);
f_gradient = @(x) 0.5 * f0_gradient(x, f_shift1, c1, d1) + 0.5 * f0_gradient(x, f_shift2, c2, d2);
f_laplacian = @(x) 0.5 * f0_laplacian(x, f_shift1, c1, d1) + 0.5 * f0_laplacian(x, f_shift2, c2, d2);


f_limit_operator_Lp = @(x) (f_laplacian(x) - p_gradient(x) .* f_gradient(x) ./ pdf(x));% ./ pdf(x).^2 ;
f_limit_operator_deltap = @(x) (f_laplacian(x) + p_gradient(x) .* f_gradient(x) ./ pdf(x));% ./ pdf(x).^2 ;

% normalize to [-1,1]
Lp_ub = max(abs(f_limit_operator_Lp(t_samp_ref)));
f_Lp = @(x) f(x) / Lp_ub;
f_gradient_Lp = @(x) f_gradient(x) / Lp_ub;
f_laplacian_Lp = @(x) f_laplacian(x) / Lp_ub;
f_limit_operator_Lp = @(x) (f_laplacian_Lp(x) - p_gradient(x) .* f_gradient_Lp(x) ./ pdf(x));

deltap_ub = max(abs(f_limit_operator_deltap(t_samp_ref)));
f_deltap = @(x) f(x) / deltap_ub;
f_gradient_deltap = @(x) f_gradient(x) / deltap_ub;
f_laplacian_deltap = @(x) f_laplacian(x) / deltap_ub;
f_limit_operator_deltap = @(x) (f_laplacian_deltap(x) + p_gradient(x) .* f_gradient_deltap(x) ./ pdf(x));



omegaM         = 5;
map_to_RD_func = @(t) L_mfd  /(2*pi) * 1/(sqrt(5))*[...
    cos(2*pi/L_mfd*t), ...
    sin(2*pi/L_mfd*t), ...
    2/omegaM*cos( 2*pi/L_mfd*omegaM*t), ...
    2/omegaM*sin( 2*pi/L_mfd*omegaM*t)];

% map_to_RD_func = @(t) L_mfd  /(2*pi) * [...
%     cos(2*pi/L_mfd*t), ...
%     sin(2*pi/L_mfd*t)];

p1 = @(x) (p_laplacian(x)./ pdf(x) + pi^2/5*(4*omegaM^2+1)) / 6;




%% kNN estimation plot


N = 2000;

x_rand_coord = sort(rand(N,1));
% x_rand_coord = linspace(0,1-1/N,N)';
x_rand_coord = interp1(y_samp_ref, t_samp_ref, x_rand_coord);
x_rand_embed = map_to_RD_func(x_rand_coord);


%
k_knn1 = 32;

rk = (k_knn1 / (alpha_d*N)).^2;
rho_bar_ref1 = zeros(N_ref,1);
for k = 1: N_ref
    x_tmp = x_ref_coord(k);
    C = [rk * p1(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_ref1(k) = abs(roots_tmp(3));
end

tic;
[knn_index, knn_d] = knnsearch(x_rand_embed, x_ref_embed, 'K', k_knn1);
rho_hat_ref1 = knn_d(:,k_knn1)/(k_knn1 / (alpha_d*N));
t_knn = toc;


k_knn2 = 64;

rk = (k_knn2 / (alpha_d*N)).^2;
rho_bar_ref2 = zeros(N_ref,1);
for k = 1: N_ref
    x_tmp = x_ref_coord(k);
    C = [rk * p1(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_ref2(k) = abs(roots_tmp(3));
end

tic;
[knn_index, knn_d] = knnsearch(x_rand_embed, x_ref_embed, 'K', k_knn2);
rho_hat_ref2 = knn_d(:,k_knn2)/(k_knn2 / (alpha_d*N));
t_knn = toc;

%
figure(310), clf;

subplot(1,2,1)
hold on
plot(x_ref_coord, rho_hat_ref1, '.', 'DisplayName', "$\hat\rho$", 'MarkerSize', 10, 'LineWidth', 3, 'Color', "#0072BD");
plot(x_ref_coord, rho_bar_ref1, 'DisplayName', "$\bar\rho_{r_k}$ (with correction)", 'MarkerSize', 3, 'LineWidth', 3, 'Color', "#D95319");
plot(x_ref_coord, 1 ./ pdf(x_ref_coord), '--', 'DisplayName', "$\bar\rho_{0}$  $\;(= p^{-1/d})$", 'MarkerSize', 3, 'LineWidth', 3, 'Color', '#EDB120');
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
plot(x_ref_coord, rho_hat_ref2, '.', 'DisplayName', "$\hat\rho$", 'MarkerSize', 10, 'LineWidth', 3, 'Color', "#0072BD");
plot(x_ref_coord, rho_bar_ref2, 'DisplayName', "$\bar\rho_{r_k}$ (with correction)", 'MarkerSize', 3, 'LineWidth', 3, 'Color', "#D95319");
plot(x_ref_coord, 1 ./ pdf(x_ref_coord), '--', 'DisplayName', "$\bar\rho_{0}$  $\;(= p^{-1/d})$", 'MarkerSize', 3, 'LineWidth', 3, 'Color', '#EDB120');
grid on
set(gca, 'FontSize', 26);
xticks(0:0.2:1);
ylim([0, 3.7])
xlabel('Intrinsic coodinate', 'Interpreter', 'latex');
% ylabel('Bandwidth functions', 'Interpreter', 'latex');
str_title = strcat('$k = ', num2str(k_knn2), '$');
title(str_title, 'Interpreter', 'latex', 'FontSize', 30);


