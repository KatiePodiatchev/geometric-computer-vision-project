close all;
addpath('laplacian\');
pause('on');

%% laod data and present it
model = load_off('./dataset/shrec/null/cat.off');
partial_model = load_off('./dataset/shrec/cuts/cuts_cat_shape_2.off');

trimesh(model.TRIV, model.VERT(:, 1), ... 
    model.VERT(:, 2), model.VERT(:, 3));
title("model");

trimesh(partial_model.TRIV, partial_model.VERT(:, 1), ... 
    partial_model.VERT(:, 2), partial_model.VERT(:, 3));
title("partial model");

%% calculate the laplacian of the shapes
global model_laplacian_W; global model_laplacian_A;
global partial_model_laplacian_W; global partial_model_laplacian_A;
[model_laplacian_W, ~, model_laplacian_A] = ...
    calc_LB_FEM_bc(model, 'dirichlet');
[partial_model_laplacian_W, ~, partial_model_laplacian_A] = ...
    calc_LB_FEM_bc(partial_model, 'dirichlet');

%% calculate the eigenvalues and vectors of the partial shape 
% laplacian
global partial_model_laplacian_eigenvalues;
[~, partial_model_laplacian_eigenvalues] = ...
    eigs(partial_model_laplacian_W, partial_model_laplacian_A, 20, 'SM');

%% find ground truth v
distances_between_vertices = pdist2(model.VERT, partial_model.VERT);
ground_truth_v = sum((distances_between_vertices < 0.1), 2);

figure()
trimesh(model.TRIV, model.VERT(:, 1), ... 
    model.VERT(:, 2), model.VERT(:, 3), ground_truth_v);
title("model with ground truth v");
rotate3d on;

%% initialize v
figure(4);
close(4);
rng(10);
bounding_box_x_min = min(model.VERT(:, 1));
bounding_box_x_max = max(model.VERT(:, 1));
bounding_box_y_min = min(model.VERT(:, 2));
bounding_box_y_max = max(model.VERT(:, 2));
bounding_box_z_min = min(model.VERT(:, 3));
bounding_box_z_max = max(model.VERT(:, 3));
bounding_box_x_length = (bounding_box_x_max - bounding_box_x_min);
bounding_box_y_length = (bounding_box_y_max - bounding_box_y_min);
bounding_box_z_length = (bounding_box_z_max - bounding_box_z_min);

mu_x = rand * 2 * bounding_box_x_length + 0.5 * bounding_box_x_min;
mu_y = rand * 2 * bounding_box_y_length + 0.5 * bounding_box_y_min;
mu_z = rand * 2 * bounding_box_z_length + 0.5 * bounding_box_z_min;

distances_between_vertices = pdist2(model.VERT, [mu_x, mu_y, mu_z]);
[mu, mu_idx] = min(distances_between_vertices);
mu = model.VERT(mu_idx, :);
tau = 1000 * diag(partial_model_laplacian_eigenvalues);
tau = tau(20);
sigma = 2 * sqrt(bounding_box_x_length * bounding_box_y_length * bounding_box_z_length);

tmp_v0 = tau * (2*pi*sigma).^(1.5) * mvnpdf(model.VERT, mu, sigma * eye(3));
v0 = min(max(tau - tmp_v0, 0), tau);

figure(4)
trimesh(model.TRIV, model.VERT(:, 1), ... 
    model.VERT(:, 2), model.VERT(:, 3), 100*v0);
title("model with initial v");
rotate3d on;

%% optimize v

v = v0;
% v = ground_truth_v;
alpha = 1e-2;
global model_laplacian_eigenvalues;
i = 0;
figure(1);
figure(2);
losses = [];
figure(10);

while true
    v = min(max(0, v - alpha *  gradient(v, 20)), tau);

    i = i + 1;
    if mod(i, 5) == 0
        figure(1);
        trimesh(model.TRIV, model.VERT(:, 1), ... 
                model.VERT(:, 2), model.VERT(:, 3), 100*v);
        title("v at iteration: " + num2str(i));
        
        figure(2)
        loss = sum(diag( ...
               (model_laplacian_eigenvalues - ...
                partial_model_laplacian_eigenvalues).^2 ./ ...
                partial_model_laplacian_eigenvalues.^2));
        losses = [losses, loss];
        plot(loss);
        title("loss at iteration: " + num2str(i));
        plot(losses);
        
        figure(3)
        plot(diag(partial_model_laplacian_eigenvalues));
        hold on;
        plot(diag(model_laplacian_eigenvalues));
        hold off;
        
        figure(4)
        subplot(1, 3, 1);
        plot(losses);
        title("loss at iteration: " + num2str(i));
        subplot(1, 2, 2);
        plot(loss);
        subplot(1, 3, 2);
        trimesh(model.TRIV, model.VERT(:, 1), ... 
        model.VERT(:, 2), model.VERT(:, 3), 100*v);
        title("v at iteration: " + num2str(i));
        subplot(1, 3, 3);
        plot(diag(partial_model_laplacian_eigenvalues));
        hold on;
        plot(diag(model_laplacian_eigenvalues));
        if i <= 7 
           initial_max_value = max(diag(model_laplacian_eigenvalues)); 
        end
        ylim([0 initial_max_value]);
        title('first 20 eigenvalues');
        legend('partial model', 'model');
        hold off;
        pause(0.0001);
    end    
end

function [gradient_v] = gradient(v, number_of_eigenvalues)
    global partial_model_laplacian_eigenvalues;
    global model_laplacian_W; global model_laplacian_A;
    global model_laplacian_eigenvalues;
    ex1 = sparse(model_laplacian_W + model_laplacian_A * diag(v));
    [eigenvectors, model_laplacian_eigenvalues] = ...
            eigs(ex1 , model_laplacian_A, number_of_eigenvalues, 'SM');
    ex2 = eigenvectors .* eigenvectors;
    ex3 = (model_laplacian_eigenvalues - ...
           partial_model_laplacian_eigenvalues) ./ ...
           partial_model_laplacian_eigenvalues .^ 2;
    gradient_v = 2 * ex2 * diag(ex3);
end
