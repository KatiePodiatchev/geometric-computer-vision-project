function [W,Sc,Sl] = calc_LB_FEM_bc(M, bc)

if bc ~= "neumann" && bc ~= "dirichlet"
    disp("options for bc are dirichlet, neumann");
end
% Stiffness (p.s.d.)

angles = zeros(M.m,3);
for i=1:3
    a = mod(i-1,3)+1;
    b = mod(i,3)+1;
    c = mod(i+1,3)+1;
    ab = M.VERT(M.TRIV(:,b),:) - M.VERT(M.TRIV(:,a),:);
    ac = M.VERT(M.TRIV(:,c),:) - M.VERT(M.TRIV(:,a),:);
    %normalize edges
    ab = ab ./ (sqrt(sum(ab.^2,2))*[1 1 1]);
    ac = ac ./ (sqrt(sum(ac.^2,2))*[1 1 1]);
    % normalize the vectors
    % compute cotan of angles
    angles(:,a) = cot(acos(sum(ab.*ac,2)));
    %cotan can also be computed by x/sqrt(1-x^2)
end

indicesI = [M.TRIV(:,1);M.TRIV(:,2);M.TRIV(:,3);M.TRIV(:,3);M.TRIV(:,2);M.TRIV(:,1)];
indicesJ = [M.TRIV(:,2);M.TRIV(:,3);M.TRIV(:,1);M.TRIV(:,2);M.TRIV(:,1);M.TRIV(:,3)];
%we add only the right angle for every vertex i 
%in the end we take W = W + W' 
%it fix it and get us the value cot(alpha)+cot(beth)
values   = [angles(:,3);angles(:,1);angles(:,2);angles(:,1);angles(:,3);angles(:,2)]*0.5;
W = sparse(indicesI, indicesJ, -values, M.n, M.n);
W = W-sparse(1:M.n,1:M.n,sum(W));

% Mass

%are of every tirangle, 
%size of all polygons!
areas = calc_tri_areas(M);

%building index for the sparse matrix,
%indexing of vertices!
indicesI = [M.TRIV(:,1);M.TRIV(:,2);M.TRIV(:,3);M.TRIV(:,3);M.TRIV(:,2);M.TRIV(:,1)];
indicesJ = [M.TRIV(:,2);M.TRIV(:,3);M.TRIV(:,1);M.TRIV(:,2);M.TRIV(:,1);M.TRIV(:,3)];

%
values   = [areas(:); areas(:); areas(:); areas(:); areas(:); areas(:)]./12;
Sc = sparse(indicesI, indicesJ, values, M.n, M.n);
Sc = Sc+sparse(1:M.n, 1:M.n, sum(Sc));

Sl = spdiag(sum(Sc,2));

if bc == "dirichlet"
    boundary = calc_boundary_edges(M.TRIV);
    boundary = unique(boundary(:));
    [W, Sc, Sl, ~] = dirichlet_bc(W, Sc, Sl, boundary, M.n, nan);
end

W = (W + W')/2;
Sc = (Sc + Sc')/2;

end
