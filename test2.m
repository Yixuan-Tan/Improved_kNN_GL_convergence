%% choose local neighborhood around x0


clear all;
close all;

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

N_samp_ref = round(L_mfd*10^5);
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

% kernerl and sigma0's
% Exponential kernel
sigma_g = 1;%0.4;
k_g_funh = @(r) exp(-r/(4*sigma_g^2)) / sqrt(4*pi*sigma_g^2) / sigma_g^2; % m_2[k] / 2 = 1
% disk kernel
r_disk = 3;
k_d_funh = @(r) ((0 <= r) & (r <= r_disk^2)) / (1/3 * r_disk^3); % m_2[h] / 2 = 1




omegaM         = 5;
map_to_RD_func = @(t) L_mfd  /(2*pi) * 1/(sqrt(5))*[...
    cos(2*pi/L_mfd*t), ...
    sin(2*pi/L_mfd*t), ...
    2/omegaM*cos( 2*pi/L_mfd*omegaM*t), ...
    2/omegaM*sin( 2*pi/L_mfd*omegaM*t)];


p1 = @(x) (p_laplacian(x)./ pdf(x) + pi^2/5*(4*omegaM^2+1)) / 6;



%%










%% set parameters

N = 40000;
k_knn = 512;
alpha_d = 2;
rk = (k_knn / (alpha_d*N)).^2;

% test point
x0_coord = 0.83;
x0_embed = map_to_RD_func(x0_coord);
epsilon_list = 10.^(-5.6:0.1:-4.2);
sigma0sq_list = (epsilon_list / rk);

% reference points to choose the two ranges
N_ref = 100;
x_ref_coord = linspace(0, 1-1/N_ref, N_ref)';
x_ref_embed = map_to_RD_func(x_ref_coord);


alpha_d = 2;
sigma0_g = 1;
sigma0_d = 1;

%% first range - sampling range

thres = 1e-5;
N_rep_samp_range = 50;

% rough sampling range
samp_lb_pre = x0_coord - 0.2;
samp_ub_pre = x0_coord + 0.2;

[inds_ref_loc, N_ref_loc] = inds_within_range(x_ref_coord, samp_lb_pre, samp_ub_pre);
x_ref_embed_loc = x_ref_embed(inds_ref_loc,:);
dismat = pdist2(x0_embed, x_ref_embed_loc, 'squaredeuclidean');

sigma0sq = sigma0sq_list(end);
dis_l_samp_list = zeros(N_rep_samp_range, 1);
dis_r_samp_list = zeros(N_rep_samp_range, 1);

t1 = tic;
for rep = 1:N_rep_samp_range

    x_rand_coord_glob = sort(rand(N,1));
    x_rand_coord_glob = interp1(y_samp_ref, t_samp_ref, x_rand_coord_glob);
    x_rand_embed_glob_knn = map_to_RD_func(x_rand_coord_glob);

    [knn_index, knn_d] = knnsearch(x_rand_embed_glob_knn, x_ref_embed_loc, 'K', k_knn);
    rho_hat_ref_loc = knn_d(:,k_knn);
    [knn_index, knn_d] = knnsearch(x_rand_embed_glob_knn, x0_embed, 'K', k_knn);
    rho_hat_x0 = knn_d(:,k_knn);

    % kernel

    dis1 = rho_hat_x0 * rho_hat_ref_loc';
    dis2 = min(repmat(rho_hat_x0, 1, N_ref_loc), rho_hat_ref_loc').^2;
    
    ker_g1 = k_g_funh( dismat ./ (sigma0sq * dis1) );
    ker_g2 = k_g_funh( dismat ./ (sigma0sq * dis2) );
    ker_d1 = k_d_funh( dismat ./ (sigma0sq * dis1) );
    ker_d2 = k_d_funh( dismat ./ (sigma0sq * dis2) );

    within_range_coord_g1 = x_ref_coord(inds_ref_loc(ker_g1 > thres));
    within_range_coord_g2 = x_ref_coord(inds_ref_loc(ker_g2 > thres));
    within_range_coord_d1 = x_ref_coord(inds_ref_loc(ker_d1 > thres));
    within_range_coord_d2 = x_ref_coord(inds_ref_loc(ker_d2 > thres));

    within_range_coord = unique([within_range_coord_g1; within_range_coord_g2; ...
        within_range_coord_d1; within_range_coord_d2]);

    [dis_l, dis_r] = distance_to_two_ends(x0_coord, within_range_coord);

    dis_l_samp_list(rep) = dis_l;
    dis_r_samp_list(rep) = dis_r;
    % if true
    %     fprintf('1');
    % end

end
t = toc(t1);
fprintf('\nFinding sampling range takes %.2f s. \n', t);


samp_lb = x0_coord - max(dis_l_samp_list);
samp_ub = x0_coord + max(dis_r_samp_list);

fprintf('\nThres = %.1e\n', thres);
fprintf('\nFound lower and upper bounds of the first range: [%.2f, %.2f]\n\n', samp_lb, samp_ub);


%% second range - knn computation range


N_rep_knn_range = 50;

[inds_ref_loc, N_ref_loc] = inds_within_range(x_ref_coord, samp_lb, samp_ub);
x_ref_embed_loc = x_ref_embed(inds_ref_loc,:);

dis_l_knn_list = zeros(N_rep_knn_range, 1);
dis_r_knn_list = zeros(N_rep_knn_range, 1);

t1 = tic;
for rep = 1:N_rep_knn_range

    x_rand_coord_glob = sort(rand(N,1));
    x_rand_coord_glob = interp1(y_samp_ref, t_samp_ref, x_rand_coord_glob);
    x_rand_embed_glob_knn = map_to_RD_func(x_rand_coord_glob);

    [knn_index, ~] = knnsearch(x_rand_embed_glob_knn, x_ref_embed_loc, 'K', k_knn);

    knn_range = x_rand_coord_glob( unique(knn_index) );
    
    [dis_l, dis_r] = distance_to_two_ends(x0_coord, knn_range);
    dis_l_knn_list(rep) = dis_l;
    dis_r_knn_list(rep) = dis_r;
    
    % if true
    %     fprintf('1');
    % end
    
end
t = toc(t1);
fprintf('\nFinding knn range takes %.2f s. \n', t);

knn_lb = x0_coord - ceil(max(dis_l_knn_list)*100)/100;
knn_ub = x0_coord + ceil(max(dis_r_knn_list)*100)/100;

fprintf('\nFound lower and upper bounds of the second range: [%.2f, %.2f]\n\n', knn_lb, knn_ub);






%%

function [inds_loc, N_loc] = inds_within_range(x_coord, range_lb, range_ub)

inds_glob = (1:length(x_coord))';
if (range_lb >= 0) && (range_ub <= 1)
    inds_loc = (x_coord <= range_ub) & (x_coord >= range_lb);
elseif range_ub > 1
    inds_loc = (x_coord <= mod(range_ub,1)) | (x_coord >= range_lb);
elseif range_lb < 0
    inds_loc = (x_coord <= range_ub) | (x_coord >= mod(range_lb,1));
end
inds_loc = inds_glob(inds_loc);
N_loc = length(inds_loc);


end



%%

function [dis_l, dis_r] = distance_to_two_ends(x0, x_ref_coord)

x0_rep = repmat(x0, 1, length(x_ref_coord));
dismat_geo = x0_rep - x_ref_coord' ;
l_inds = (  ((dismat_geo>=0) & (dismat_geo<0.5))  &  x0_rep > 0.5  ) ...
        | (  ((dismat_geo>=0) | (dismat_geo<-0.5))  &  x0_rep <= 0.5  );
r_inds = (  ((dismat_geo<=0) & (dismat_geo>-0.5))  &  x0_rep < 0.5  ) ...
        | (  ((dismat_geo<=0) | (dismat_geo>0.5))  &  x0_rep >= 0.5  );
dismat_geo = abs(dismat_geo);
dismat_geo = min(dismat_geo, 1-dismat_geo);

dis_l = max( dismat_geo .*  l_inds);
dis_r = max( dismat_geo .*  r_inds);

end








