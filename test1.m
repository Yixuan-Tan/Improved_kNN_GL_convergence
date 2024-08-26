%% graph Laplacian point-wise convergence
% isometric embedding of S^1 in R^4


clear all;
close all;

%%

rng("shuffle");

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
r_disk = 3;
k_d_funh = @(r) ((0 <= r) & (r <= r_disk^2)) / (1/3 * r_disk^3); % m_2[h] / 2 = 1


% f
f0 = @(x, f_shift, c, d) sin(2*pi/L_mfd*c*(x-f_shift)) * d;
f0_gradient = @(x, f_shift, c, d) 2*pi/L_mfd*c*cos(2*pi/L_mfd*c*(x-f_shift)) * d;
f0_laplacian = @(x, f_shift, c, d) -(2*pi/L_mfd)^2*c^2*sin(2*pi/L_mfd*c*(x-f_shift)) * d;

f_shift1 = 0.1;  c1 = 1; d1 = 1;
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

p1 = @(x) (p_laplacian(x)./ pdf(x) + pi^2/5*(4*omegaM^2+1)) / 6;



%%



N_ref = 100;


sigma0sq_list = 10.^(-1.2:0.1:0.2);
k_knn = 512; alpha_d = 2; N_pre = 40000;
rk = (k_knn / (alpha_d*N_pre)).^2;
epsilon_list = sigma0sq_list * rk;

% evaluation points
x_ref_coord = linspace(0,1-1/N_ref,N_ref)';
x_ref_embed = map_to_RD_func(x_ref_coord);

loc_ind = 84;


return



%% Bias error

N_sigma = length(sigma0sq_list);


ind_ref_bias = (1:100)';
N_ref_bias = length(ind_ref_bias);
x_ref_coord_bias = x_ref_coord(ind_ref_bias);
x_ref_embed_bias = map_to_RD_func(x_ref_coord_bias);


N_integral = 256000;
x_determ_coord = linspace(0, L_mfd*(1-1/N_integral), N_integral)';
x_determ_embed = map_to_RD_func(x_determ_coord);
rho_bar_determ = zeros(N_integral,1);

N_integral2 = 2*N_integral;
x_determ_coord2 = linspace(0, L_mfd*(1-1/N_integral2), N_integral2)';
x_determ_embed2 = map_to_RD_func(x_determ_coord2);
rho_bar_determ2 = zeros(N_integral2,1);


% aff_type = "un-norm";
aff_type = "norm";

% limiting operator
if strcmpi(aff_type, "un-norm")
    f_vec_determ = f_Lp(x_ref_coord_bias);
    deltaf_limit_ref_bias = f_limit_operator_Lp(x_ref_coord_bias);
elseif strcmpi(aff_type, "norm")
    f_vec_determ = f_deltap(x_ref_coord_bias);
    deltaf_limit_ref_bias = f_limit_operator_deltap(x_ref_coord_bias);
end

dismat1   = pdist2(x_ref_embed_bias,x_determ_embed, 'fastsquaredeuclidean');
dismat2   = pdist2(x_ref_embed_bias,x_determ_embed2, 'fastsquaredeuclidean');


%%
est_bias_mat_g1  = zeros(N_ref_bias, N_sigma);
est_bias_mat_g2  = zeros(N_ref_bias, N_sigma);
est_bias_mat_g3  = zeros(N_ref_bias, N_sigma);
est_bias_mat_d1  = zeros(N_ref_bias, N_sigma);
est_bias_mat_d2  = zeros(N_ref_bias, N_sigma);

est_bias_mat_g1_2  = zeros(N_ref_bias, N_sigma);
est_bias_mat_g2_2  = zeros(N_ref_bias, N_sigma);
est_bias_mat_g3_2  = zeros(N_ref_bias, N_sigma);
est_bias_mat_d1_2  = zeros(N_ref_bias, N_sigma);
est_bias_mat_d2_2  = zeros(N_ref_bias, N_sigma);


%%

% true \bar{\rho} for X
rho_bar_ref_determ = zeros(N_ref_bias,1);
for k = 1: N_ref_bias
    x_tmp = x_ref_coord_bias(k);
    C = [rk * p1(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_ref_determ(k) = abs(roots_tmp(end));
end
% toc;

for k = 1: N_integral
    x_tmp = x_determ_coord(k);
    C = [rk * p1(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_determ(k) = abs(roots_tmp(end));
end
for k = 1: N_integral2
    x_tmp = x_determ_coord2(k);
    C = [rk * p1(x_tmp) 0 1 -1/pdf(x_tmp)];
    roots_tmp = roots(C);
    rho_bar_determ2(k) = abs(roots_tmp(end));
end
%
t_start = tic;
for i = 1: N_sigma

    epsilon = epsilon_list(i);
    sigma0sq = sigma0sq_list(i);

    fprintf('\n    %d-th sigma_0^2: 10^(%.1f)\n', i, log10(sigma0sq));

    %% integrated GL; four kernels; analytical distance (\bar{\rho})


    p_vec = pdf(x_determ_coord);
    if strcmpi(aff_type, "un-norm")
        f_vec = f_Lp(x_determ_coord);
    elseif strcmpi(aff_type, "norm")
        f_vec = f_deltap(x_determ_coord);
    end
    dis1 = rho_bar_ref_determ * rho_bar_determ';
    dis2 = min(repmat(rho_bar_ref_determ, 1, N_integral), repmat(rho_bar_determ', N_ref_bias, 1)).^2;
    dis3 = (rho_bar_ref_determ.^2 + (rho_bar_determ').^2)/2;

    if strcmpi(aff_type, "un-norm")
        ker_g1_determ = k_g_funh(  dismat1 ./ (epsilon * dis1)  );
        ker_g2_determ = k_g_funh(  dismat1 ./ (epsilon * dis2)  );
        ker_g3_determ = k_g_funh(  dismat1 ./ (epsilon * dis3)  );
        ker_d1_determ = k_d_funh(  dismat1 ./ (epsilon * dis1)  );
        ker_d2_determ = k_d_funh(  dismat1 ./ (epsilon * dis2)  );
    elseif strcmpi(aff_type, "norm")
        ker_g1_determ = k_g_funh(  dismat1 ./ (epsilon * dis1)  ) ./ dis1;
        ker_g2_determ = k_g_funh(  dismat1 ./ (epsilon * dis2)  ) ./ dis2;
        ker_g3_determ = k_g_funh(  dismat1 ./ (epsilon * dis3)  ) ./ dis3;
        ker_d1_determ = k_d_funh(  dismat1 ./ (epsilon * dis1)  ) ./ dis1;
        ker_d2_determ = k_d_funh(  dismat1 ./ (epsilon * dis2)  ) ./ dis2;
    end

    Lf_g1_determ = 1 / (sigma_g^2 * epsilon) * ((ker_g1_determ * (p_vec.*f_vec)) ./ (ker_g1_determ * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_g2_determ = 1 / (sigma_g^2 * epsilon) * ((ker_g2_determ * (p_vec.*f_vec)) ./ (ker_g2_determ * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_g3_determ = 1 / (sigma_g^2 * epsilon) * ((ker_g3_determ * (p_vec.*f_vec)) ./ (ker_g3_determ * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_d1_determ = 1 / (r_disk^2/6  * epsilon) * ((ker_d1_determ * (p_vec.*f_vec)) ./ (ker_d1_determ * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_d2_determ = 1 / (r_disk^2/6  * epsilon) * ((ker_d2_determ * (p_vec.*f_vec)) ./ (ker_d2_determ * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);

    %% compute errors
    % --------------------------------------------------------------------------

    est_bias_mat_g1(:, i)  = Lf_g1_determ;
    est_bias_mat_g2(:, i)  = Lf_g2_determ;
    est_bias_mat_g3(:, i)  = Lf_g3_determ;
    est_bias_mat_d1(:, i)  = Lf_d1_determ;
    est_bias_mat_d2(:, i)  = Lf_d2_determ;


    %% integrated GL; four kernels; analytical distance (\bar{\rho})


    p_vec = pdf(x_determ_coord2);
    if strcmpi(aff_type, "un-norm")
        f_vec = f_Lp(x_determ_coord2);
    elseif strcmpi(aff_type, "norm")
        f_vec = f_deltap(x_determ_coord2);
    end
    dis1 = rho_bar_ref_determ * rho_bar_determ2';
    dis2 = min(repmat(rho_bar_ref_determ, 1, N_integral2), repmat(rho_bar_determ2', N_ref_bias, 1)).^2;
    dis3 = (rho_bar_ref_determ.^2 + (rho_bar_determ2').^2)/2;

    if strcmpi(aff_type, "un-norm")
        ker_g1_determ2 = k_g_funh(  dismat2 ./ (epsilon * dis1)  );
        ker_g2_determ2 = k_g_funh(  dismat2 ./ (epsilon * dis2)  );
        ker_g3_determ3 = k_g_funh(  dismat2 ./ (epsilon * dis3)  );
        ker_d1_determ2 = k_d_funh(  dismat2 ./ (epsilon * dis1)  );
        ker_d2_determ2 = k_d_funh(  dismat2 ./ (epsilon * dis2)  );
    elseif strcmpi(aff_type, "norm")
        ker_g1_determ2 = k_g_funh(  dismat2 ./ (epsilon * dis1)  ) ./ dis1;
        ker_g2_determ2 = k_g_funh(  dismat2 ./ (epsilon * dis2)  ) ./ dis2;
        ker_g3_determ2 = k_g_funh(  dismat2 ./ (epsilon * dis3)  ) ./ dis3;
        ker_d1_determ2 = k_d_funh(  dismat2 ./ (epsilon * dis1)  ) ./ dis1;
        ker_d2_determ2 = k_d_funh(  dismat2 ./ (epsilon * dis2)  ) ./ dis2;
    end

    Lf_g1_determ2 = 1 / (sigma_g^2 * epsilon) * ((ker_g1_determ2 * (p_vec.*f_vec)) ./ (ker_g1_determ2 * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_g2_determ2 = 1 / (sigma_g^2 * epsilon) * ((ker_g2_determ2 * (p_vec.*f_vec)) ./ (ker_g2_determ2 * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_g3_determ2 = 1 / (sigma_g^2 * epsilon) * ((ker_g3_determ2 * (p_vec.*f_vec)) ./ (ker_g3_determ2 * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_d1_determ2 = 1 / (r_disk^2/6  * epsilon) * ((ker_d1_determ2 * (p_vec.*f_vec)) ./ (ker_d1_determ2 * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);
    Lf_d2_determ2 = 1 / (r_disk^2/6  * epsilon) * ((ker_d2_determ2 * (p_vec.*f_vec)) ./ (ker_d2_determ2 * p_vec) - f_vec_determ) .* rho_bar_ref_determ.^(-2);

    %% compute errors
    % --------------------------------------------------------------------------

    est_bias_mat_g1_2(:, i)  = Lf_g1_determ2;
    est_bias_mat_g2_2(:, i)  = Lf_g2_determ2;
    est_bias_mat_g3_2(:, i)  = Lf_g3_determ2;
    est_bias_mat_d1_2(:, i)  = Lf_d1_determ2;
    est_bias_mat_d2_2(:, i)  = Lf_d2_determ2;

    t1 = toc(t_start);

    fprintf('Currently takes %.2f s; Remaining time %.2fs\n', t1, t1/i *( N_sigma-i ) );

end






t2 = toc(t_start);
fprintf('\nTotally takes %.2f s\n', t2 );



%%


err_bias_g1 = abs(est_bias_mat_g1_2(loc_ind,:) - deltaf_limit_ref_bias(loc_ind));
err_bias_g2 = abs(est_bias_mat_g2_2(loc_ind,:) - deltaf_limit_ref_bias(loc_ind));
err_bias_g3 = abs(est_bias_mat_g3_2(loc_ind,:) - deltaf_limit_ref_bias(loc_ind));
err_bias_d1 = abs(est_bias_mat_d1_2(loc_ind,:) - deltaf_limit_ref_bias(loc_ind));
err_bias_d2 = abs(est_bias_mat_d2_2(loc_ind,:) - deltaf_limit_ref_bias(loc_ind));
thres = 0.2;
flag_g1 = abs(est_bias_mat_g1_2(loc_ind,:) - est_bias_mat_g1(loc_ind,:)) < thres * err_bias_g1;
flag_g2 = abs(est_bias_mat_g2_2(loc_ind,:) - est_bias_mat_g2(loc_ind,:)) < thres * err_bias_g2;
flag_g3 = abs(est_bias_mat_g3_2(loc_ind,:) - est_bias_mat_g3(loc_ind,:)) < thres * err_bias_g3;
flag_d1 = abs(est_bias_mat_d1_2(loc_ind,:) - est_bias_mat_d1(loc_ind,:)) < thres * err_bias_d1;
flag_d2 = abs(est_bias_mat_d2_2(loc_ind,:) - est_bias_mat_d2(loc_ind,:)) < thres * err_bias_d2;

flag_g1 = logical(fliplr(cumprod(fliplr(flag_g1)))); flag_g2 = logical(fliplr(cumprod(fliplr(flag_g2)))); flag_g3 = logical(fliplr(cumprod(fliplr(flag_g3))));
flag_d1 = logical(fliplr(cumprod(fliplr(flag_d1)))); flag_d2 = logical(fliplr(cumprod(fliplr(flag_d2))));








%% repeat for multiple times, a list of sigma0



N_rep = 2000;


sample_range_lb = 0.76; % 0; %
sample_range_ub = 0.98; % 0.7; % 0.55; % 1; %

knn_lb = 0.75;
knn_ub = 0.99;


inds_glob = (1:N_pre)';
x_ref_coord_loc = x_ref_coord(loc_ind);
x_ref_embed_loc = x_ref_embed(loc_ind,:);
N_ref_loc = length(loc_ind);



%%

est_L_g1_mat = zeros(N_ref_loc, N_sigma, N_rep);
est_L_g2_mat = zeros(N_ref_loc, N_sigma, N_rep);
est_L_g3_mat = zeros(N_ref_loc, N_sigma, N_rep);
est_L_d1_mat = zeros(N_ref_loc, N_sigma, N_rep);
est_L_d2_mat = zeros(N_ref_loc, N_sigma, N_rep);


% aff_type = "un-norm";
aff_type = "norm";

% limiting operator
if strcmpi(aff_type, "un-norm")
    f_vec_loc = f_Lp(x_ref_coord_loc);
    deltaf_limit_ref_loc = f_limit_operator_Lp(x_ref_coord_loc);
elseif strcmpi(aff_type, "norm")
    f_vec_loc = f_deltap(x_ref_coord_loc);
    deltaf_limit_ref_loc = f_limit_operator_deltap(x_ref_coord_loc);
end


t_start = tic;

for i = 1: N_sigma

    fprintf('\n%d-th sigma0 \n', i);

    sigma0sq = sigma0sq_list(i);


    tic;

    %%
    for rep = 1: N_rep

        x_rand_coord_glob = sort(rand(N_pre,1));
        x_rand_coord_glob = interp1(y_samp_ref, t_samp_ref, x_rand_coord_glob);

        if (sample_range_lb >= 0) && (sample_range_ub <= 1)
            inds_loc = (x_rand_coord_glob <= sample_range_ub) & (x_rand_coord_glob >= sample_range_lb);
        elseif sample_range_ub > 1
            inds_loc = (x_rand_coord_glob <= mod(sample_range_ub,1)) | (x_rand_coord_glob >= sample_range_lb);
        elseif sample_range_lb < 0
            inds_loc = (x_rand_coord_glob <= sample_range_ub) | (x_rand_coord_glob >= mod(sample_range_lb,1));
        end
        inds_loc = inds_glob(inds_loc);
        N_loc = length(inds_loc);
        % sample X (local)
        x_rand_coord_loc = x_rand_coord_glob(inds_loc);
        x_rand_embed_loc = map_to_RD_func(x_rand_coord_loc);

        if (knn_lb >= 0) && (knn_ub <= 1)
            knn_inds_loc = (x_rand_coord_glob <= knn_ub) & (x_rand_coord_glob >= knn_lb);
        elseif knn_ub > 1
            knn_inds_loc = (x_rand_coord_glob <= mod(knn_ub,1)) | (x_rand_coord_glob >= knn_lb);
        elseif knn_lb < 0
            knn_inds_loc = (x_rand_coord_glob <= knn_ub) | (x_rand_coord_glob >= mod(knn_lb,1));
        end
        knn_inds_loc = inds_glob(knn_inds_loc);
        x_rand_embed_glob_knn = map_to_RD_func(x_rand_coord_glob(knn_inds_loc));


        tic;
        [knn_index, knn_d] = knnsearch(x_rand_embed_glob_knn, x_ref_embed_loc, 'K', k_knn);
        Rhat_ref_loc = knn_d(:,k_knn);
        [knn_index, knn_d] = knnsearch(x_rand_embed_glob_knn, x_rand_embed_loc, 'K', k_knn);
        Rhat_X_loc = knn_d(:,k_knn);
        t_knn = toc;
        % fprintf('\nt_knn: %.4f s\n', t_knn);


        tic;
        dismat = pdist2(x_ref_embed_loc, x_rand_embed_loc, 'fastsquaredeuclidean');
        t_dis = toc;
        % fprintf('\nt_dis: %.4f s\n', t_dis);
        %%
        % f valued vector for X
        if strcmpi(aff_type, "un-norm")
            f_vec_X = f_Lp(x_rand_coord_loc);
        elseif strcmpi(aff_type, "norm")
            f_vec_X = f_deltap(x_rand_coord_loc);
        end

        % kernel matrices & GL; five kernels
        tic;
        dis_g1 = Rhat_ref_loc * Rhat_X_loc';
        dis_g2 = min(repmat(Rhat_ref_loc, 1, N_loc), repmat(Rhat_X_loc', N_ref_loc, 1)).^2;
        dis_g3 = (Rhat_ref_loc.^2 + (Rhat_X_loc').^2)/2;
        dis_d1 = Rhat_ref_loc * Rhat_X_loc';
        dis_d2 = min(repmat(Rhat_ref_loc, 1, N_loc), repmat(Rhat_X_loc', N_ref_loc, 1)).^2;
        % compute kernels
        ker_g1 = k_g_funh(  dismat ./ (sigma0sq * dis_g1)  );
        ker_g2 = k_g_funh(  dismat ./ (sigma0sq * dis_g2)  );
        ker_g3 = k_g_funh(  dismat ./ (sigma0sq * dis_g3)  );
        ker_d1 = k_d_funh(  dismat ./ (sigma0sq * dis_d1)  );
        ker_d2 = k_d_funh(  dismat ./ (sigma0sq * dis_d2)  );
        % 
        if strcmpi(aff_type, "un-norm")
            ker_g1_X = ker_g1;
            ker_g2_X = ker_g2;
            ker_g3_X = ker_g3;
            ker_d1_X = ker_d1;
            ker_d2_X = ker_d2;
        elseif strcmpi(aff_type, "norm")
            ker_g1_X = ker_g1 ./ dis_g1;
            ker_g2_X = ker_g2 ./ dis_g2;
            ker_g3_X = ker_g3 ./ dis_g3;
            ker_d1_X = ker_d1 ./ dis_d1;
            ker_d2_X = ker_d2 ./ dis_d2;
        end
        t_ker = toc;
        % fprintf('\nt_ker: %.4f s\n', t_ker);
        tic;
        % compute GLs
        Lf_g1 = 1 / (sigma_g^2 * sigma0sq) * ( (ker_g1_X * f_vec_X) ./ sum(ker_g1_X, 2) - f_vec_loc) .* Rhat_ref_loc.^(-2);
        Lf_g2 = 1 / (sigma_g^2 * sigma0sq) * ( (ker_g2_X * f_vec_X) ./ sum(ker_g2_X, 2) - f_vec_loc) .* Rhat_ref_loc.^(-2);
        Lf_g3 = 1 / (sigma_g^2 * sigma0sq) * ( (ker_g3_X * f_vec_X) ./ sum(ker_g3_X, 2) - f_vec_loc) .* Rhat_ref_loc.^(-2);
        Lf_d1 = 1 / (r_disk^2/6  * sigma0sq) * ( (ker_d1_X * f_vec_X) ./ sum(ker_d1_X, 2) - f_vec_loc) .* Rhat_ref_loc.^(-2);
        Lf_d2 = 1 / (r_disk^2/6  * sigma0sq) * ( (ker_d2_X * f_vec_X) ./ sum(ker_d2_X, 2) - f_vec_loc) .* Rhat_ref_loc.^(-2);
        t_gl = toc;
        % fprintf('t_GL: %.4f s\n', t_gl);


        est_L_g1_mat(:,i,rep) = Lf_g1;
        est_L_g2_mat(:,i,rep) = Lf_g2;
        est_L_g3_mat(:,i,rep) = Lf_g3;
        est_L_d1_mat(:,i,rep) = Lf_d1;
        est_L_d2_mat(:,i,rep) = Lf_d2;



        if mod(rep, 100) == 0
            fprintf('        Have run %2d replicas\n', rep);
        end


    end

    t1 = toc(t_start);
    fprintf('    Currently takes %.2f s; Remaining time %.2fs\n', t1, t1/i *( N_sigma-i ) );
    fprintf('    ------------------------------------------------------\n');
    fprintf('\n    L\n');
    fprintf('    Abso Err of Exp  + self-tuned: %.4f\n', mean(abs(est_L_g1_mat(1,i,:) - deltaf_limit_ref_loc(1))));
    fprintf('    Abso Err of Exp  + min       : %.4f\n', mean(abs(est_L_g2_mat(1,i,:) - deltaf_limit_ref_loc(1))));
    fprintf('    Abso Err of Disk + self-tuned: %.4f\n', mean(abs(est_L_d1_mat(1,i,:) - deltaf_limit_ref_loc(1))));
    fprintf('    Abso Err of Disk + min       : %.4f\n', mean(abs(est_L_d2_mat(1,i,:) - deltaf_limit_ref_loc(1))));
    fprintf('    ------------------------------------------------------\n');


end







%%
color1 = "#D95319";
color2 = "#77AC30"; % color2 = "#FE6767";
color3 = "#7E2F8E";
color4 = "#0072BD";
color5 = "#EDB120";


%% Figure 2 in the paper

plt_ind = 1;

[~, plt_ind_bias] = min(abs(x_ref_coord_bias - x_ref_coord_loc(plt_ind)))


err_g1 = mean(abs(est_L_g1_mat(plt_ind,:,:) - deltaf_limit_ref_loc(plt_ind)), 3);
err_g3 = mean(abs(est_L_g3_mat(plt_ind,:,:) - deltaf_limit_ref_loc(plt_ind)), 3);
figure(2009), clf;
subplot(1,2,1);
hold on; grid on;
plot(log10(sigma0sq_list), log10(  err_g1  ) , 'd-', 'MarkerSize', 10, 'Color', color1, 'LineWidth', 3, 'DisplayName', '$(i)$  $k_0 = \exp(-\eta/4)$, self-tuned $\phi$')
plot(log10(sigma0sq_list), log10(  err_g3  ) , 'd-', 'MarkerSize', 10, 'Color', color5, 'LineWidth', 3, 'DisplayName', '$(ii)$ $k_0 = \exp(-\eta/4)$, squared-mean $\phi$')
legend('Location', 'best', 'Interpreter', 'latex', 'FontSize', 26);
L = legend; L.AutoUpdate = 'off';
plot(log10(sigma0sq_list(flag_g1)), log10(err_bias_g1(flag_g1)), '--', 'Color', color1, 'LineWidth', 3);
plot(log10(sigma0sq_list(flag_g3)), log10(err_bias_g3(flag_g3)), '--', 'Color', color5, 'LineWidth', 3);

i_min = 5; i_max = min(15, length(log10(sigma0sq_list(flag_g1))));
x = log10(sigma0sq_list(flag_g1)); y = log10(err_bias_g1(flag_g1));
p_g1 = polyfit(x(i_min:i_max), y(i_min:i_max), 1);
i_min = 4; i_max = min(15, length(log10(sigma0sq_list(flag_g3))));
x = log10(sigma0sq_list(flag_g3)); y = log10(err_bias_g3(flag_g3));
p_g3 = polyfit(x(i_min:i_max), y(i_min:i_max), 1);
plot(log10(sigma0sq_list), polyval(p_g1, log10(sigma0sq_list)), '-.', 'Color', color1, 'LineWidth', 2);
plot(log10(sigma0sq_list), polyval(p_g3, log10(sigma0sq_list)), '-.', 'Color', color5, 'LineWidth', 2);

x = [0.37, 0.34]; y = [0.24, 0.29]; pos = [0.4, 0.2, 0.01, 0.01];
str = strcat('slope = $', num2str(p_g1(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color1, 'Interpreter', 'latex', 'TextColor', color1);
x = [0.37, 0.3]; y = [0.18, 0.3]; pos = [0.4, 0.2, 0.01, 0.01];
str = strcat('slope = $', num2str(p_g3(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color5, 'Interpreter', 'latex', 'TextColor', color5);

i_min = 1; i_max = 3; x = log10(sigma0sq_list(i_min:i_max)); y = log10(err_g1(i_min:i_max));
p_g1_var = polyfit(x, y, 1);
i_min = 1; i_max = 3; x = log10(sigma0sq_list(i_min:i_max)); y = log10(err_g3(i_min:i_max));
p_g3_var = polyfit(x, y, 1);
i_plt_max = 12;
plot(log10(sigma0sq_list(1:i_plt_max)), polyval(p_g1_var, log10(sigma0sq_list(1:i_plt_max))), '-.', 'Color', color1, 'LineWidth', 2);
plot(log10(sigma0sq_list(1:i_plt_max)), polyval(p_g3_var, log10(sigma0sq_list(1:i_plt_max))), '-.', 'Color', color5, 'LineWidth', 2);

x = [0.24, 0.28]; y = [0.4, 0.46]; pos = [0.4, 0.2, 0.01, 0.01];
str = strcat('slope = $', num2str(p_g1_var(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color1, 'Interpreter', 'latex', 'TextColor', color1);
x = [0.24, 0.29]; y = [0.34, 0.43]; pos = [0.4, 0.2, 0.01, 0.01];
str = strcat('slope = $', num2str(p_g3_var(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color5, 'Interpreter', 'latex', 'TextColor', color5);

set(gca, 'Fontsize', 26);
% title({'', 'Fast-rate kernels'}, 'Interpreter', 'latex', 'FontSize', 26)
xlim([log10(sigma0sq_list(1)), log10(sigma0sq_list(N_sigma))]);
ylim([-2, -0.5])
xlabel('$\log_{10} \sigma_0^2$', 'Interpreter', 'latex', 'FontSize', 26);
ylabel('$\log_{10}$ Err', 'Interpreter', 'latex', 'FontSize', 26);





err_g2 = mean(abs(est_L_g2_mat(plt_ind,:,:) - deltaf_limit_ref_loc(plt_ind)), 3);
err_d1 = mean(abs(est_L_d1_mat(plt_ind,:,:) - deltaf_limit_ref_loc(plt_ind)), 3);
err_d2 = mean(abs(est_L_d2_mat(plt_ind,:,:) - deltaf_limit_ref_loc(plt_ind)), 3);
subplot(1,2,2);
hold on; grid on;
plot(log10(sigma0sq_list), log10(  err_g2  ) , 'd-', 'MarkerSize', 10, 'Color', color2, 'LineWidth', 3, 'DisplayName', '$(iii)$ $k_0 = \exp(-\eta/4)$, min $\phi$')
plot(log10(sigma0sq_list), log10(  err_d1  ) , 's-', 'MarkerSize', 10, 'Color', color3, 'LineWidth', 3, 'DisplayName', '$(iv)$  $k_0 = \mathbf{1}_{[0,3]}(\eta)$, self-tuned $\phi$')
plot(log10(sigma0sq_list), log10(  err_d2  ) , 's-', 'MarkerSize', 10, 'Color', color4, 'LineWidth', 3, 'DisplayName', '$(v)$   $k_0 = \mathbf{1}_{[0,3]}(\eta)$, min $\phi$')
legend('Location', 'northeast', 'Interpreter', 'latex', 'FontSize', 26);
L = legend; L.AutoUpdate = 'off';
plot(log10(sigma0sq_list(flag_g2)), log10(err_bias_g2(flag_g2)), '--', 'Color', color2, 'LineWidth', 3);
plot(log10(sigma0sq_list(flag_d1)), log10(err_bias_d1(flag_d1)), '--', 'Color', color3, 'LineWidth', 3);
plot(log10(sigma0sq_list(flag_d2)), log10(err_bias_d2(flag_d2)), '--', 'Color', color4, 'LineWidth', 3);

i_min = 1; i_max = min(15, length(log10(sigma0sq_list(flag_g2))));
x = log10(sigma0sq_list(flag_g2)); y = log10(err_bias_g2(flag_g2));
p_g2 = polyfit(x(i_min:i_max), y(i_min:i_max), 1);
i_min = 7; i_max = min(15, length(log10(sigma0sq_list(flag_d1))));
x = log10(sigma0sq_list(flag_d1)); y = log10(err_bias_d1(flag_d1));
p_d1 = polyfit(x(i_min:i_max), y(i_min:i_max), 1);
i_min = 1; i_max = min(15, length(log10(sigma0sq_list(flag_d2))));
x = log10(sigma0sq_list(flag_d2)); y = log10(err_bias_d2(flag_d2));
p_d2 = polyfit(x(i_min:i_max), y(i_min:i_max), 1);
plot(log10(sigma0sq_list), polyval(p_g2, log10(sigma0sq_list)), '-.', 'Color', color2, 'LineWidth', 2);
plot(log10(sigma0sq_list), polyval(p_d1, log10(sigma0sq_list)), '-.', 'Color', color3, 'LineWidth', 2);
plot(log10(sigma0sq_list), polyval(p_d2, log10(sigma0sq_list)), '-.', 'Color', color4, 'LineWidth', 2);

x = [0.66, 0.64]; y = [0.32, 0.39]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_g2(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color2, 'Interpreter', 'latex', 'TextColor', color2);
x = [0.81, 0.79]; y = [0.16, 0.22]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_d1(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color3, 'Interpreter', 'latex', 'TextColor', color3);
x = [0.66, 0.63]; y = [0.26, 0.38]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_d2(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color4, 'Interpreter', 'latex', 'TextColor', color4);

i_min = 1; i_max = 3; x = log10(sigma0sq_list(i_min:i_max)); y = log10(err_g2(i_min:i_max));
p_g2_var = polyfit(x, y, 1);
i_min = 2; i_max = 4; x = log10(sigma0sq_list(i_min:i_max)); y = log10(err_d1(i_min:i_max));
p_d1_var = polyfit(x, y, 1)
i_min = 2; i_max = 4; x = log10(sigma0sq_list(i_min:i_max)); y = log10(err_d2(i_min:i_max));
p_d2_var = polyfit(x, y, 1);
i_plt_max = 12;
plot(log10(sigma0sq_list(1:i_plt_max)), polyval(p_g2_var, log10(sigma0sq_list(1:i_plt_max))), '-.', 'Color', color2, 'LineWidth', 2);
plot(log10(sigma0sq_list(1:i_plt_max)), polyval(p_d1_var, log10(sigma0sq_list(1:i_plt_max))), '-.', 'Color', color3, 'LineWidth', 2);
plot(log10(sigma0sq_list(1:i_plt_max)), polyval(p_d2_var, log10(sigma0sq_list(1:i_plt_max))), '-.', 'Color', color4, 'LineWidth', 2);

x = [0.7, 0.65]; y = [0.67, 0.59]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_g2_var(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color2, 'Interpreter', 'latex', 'TextColor', color2);
x = [0.64, 0.66]; y = [0.5, 0.58]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_d1_var(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color3, 'Interpreter', 'latex', 'TextColor', color3);
x = [0.7, 0.68]; y = [0.61, 0.57]; pos = [0.62 0.27 0.1 0.1];
str = strcat('slope = $', num2str(p_d2_var(1), '%.2f'), '$');
annotation('textarrow', x, y, 'String', str, 'FontSize', 26, 'Color', color4, 'Interpreter', 'latex', 'TextColor', color4);

set(gca, 'Fontsize', 26);
% title({'', 'Slow-rate kernels'}, 'Interpreter', 'latex', 'FontSize', 26)
xlim([log10(sigma0sq_list(1)), log10(sigma0sq_list(N_sigma))]);
ylim([-2, -0.5])
xlabel('$\log_{10} \sigma_0^2$', 'Interpreter', 'latex', 'FontSize', 26);
ylabel('$\log_{10}$ Err', 'Interpreter', 'latex', 'FontSize', 26);

sgtitle({""}, 'Interpreter', 'latex', 'FontSize', 30);








