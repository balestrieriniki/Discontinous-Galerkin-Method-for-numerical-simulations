
clear

n = 20;         % number of elements
Tf = 0.2;         % final time
boundary = 1;   % switch between Dirichlet conditions (1) and Absorbing (0)
k = 2;          % polynomial degree
c=1;            % avection term, light velocity
rho=1;          % density
Cr = 1.5;       % Courant number -> sets time step size in dt = Cr * h / a
flux = 1;       % flux type, 0 = LF, 1 = HDG
nc = k+1;       % number of quadrature points
left = 0;       % left end of the domain
right = 1;      % right end of the domain
plot_accurate = 1; % plot only on nodes (0) or with more resolution (1)

%% ANALYTICAL SOLUTION
% We define two for velocity and pressure
analytical_p = @(x, t) sin(pi * x) * sin(pi * t);
analytical_v = @(x, t) cos(pi * x) * cos(pi * t);

% set quadrature formula and quadrature nodes for integration
[pg_gauss_lobatto,wg_gauss_lobatto] = get_gauss_lobatto_quadrature(nc);
[pg_gauss,wg_gauss] = get_gauss_quadrature(nc);

% set the node points for the Lagrange polynomials
xunit = get_gauss_lobatto_quadrature(k+1);

%% START THE SIMULATION

% create mesh y and compute mesh size h
kp1 = k+1;
y = zeros(2*n,1);
h = zeros(n,1);
for e=1:n
    y(2*e-1) = left + (right-left)*(e-1)/n;
    y(2*e) = left + (right-left)*e/n;
    h(e) = y(2*e)-y(2*e-1);
end

% create the grid of all node points for interpolation
for e=1:n
    x((kp1*e-k):kp1*e) = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*xunit);
end

% compute time step from Cr number, adjust to hit the desired final time
dt = Cr * min(h) / (c * (k^1.5));
NT = round(Tf/dt);
dt = Tf/NT;

%% MASS MATRIX FOR GAUSS
[values_gauss,derivatives_gauss] = evaluate_lagrange_basis(xunit, pg_gauss);
Me_gauss = values_gauss * diag(wg_gauss) * values_gauss'; 
M1inv_gauss = sparse(kp1*n,kp1*n);

for e=1:n
    M1inv_gauss((kp1*e-k):kp1*e,(kp1*e-k):kp1*e) = inv(0.5*h(e)*Me_gauss);
end
Minv_gauss = blkdiag(M1inv_gauss, M1inv_gauss);

%% MASS MATRIX FOR GAUSS-LOBATTO
[values_gauss_lobatto,derivatives_gauss_lobatto] = evaluate_lagrange_basis(xunit, pg_gauss_lobatto);
Me_gauss_lobatto = values_gauss_lobatto * diag(wg_gauss_lobatto) * values_gauss_lobatto'; 
M1inv_gauss_lobatto = sparse(kp1*n,kp1*n);

for e=1:n
    M1inv_gauss_lobatto((kp1*e-k):kp1*e,(kp1*e-k):kp1*e) = inv(0.5*h(e)*Me_gauss_lobatto);
end
Minv_gauss_lobatto = blkdiag(M1inv_gauss_lobatto, M1inv_gauss_lobatto);

%% S^T - F MATRIX FOR GAUSS
% Calculate (S^T - F) matrix by calling evaluate_wave_rhs on the vectors of
% the canonical base of R^(2n*nk1)
ST_minus_F_gauss = sparse(2*kp1*n,2*kp1*n);
for j = 1 : 2*kp1*n
    base_vector = sparse(2*kp1*n, 1);
    base_vector(j) = 1;
    matrix_column = evaluate_wave_rhs(base_vector, c, rho, [0,0], values_gauss, derivatives_gauss, wg_gauss, flux, boundary);
    ST_minus_F_gauss(:, j) = matrix_column;
    
end
eigenvalues_gauss = eig(full(Minv_gauss*ST_minus_F_gauss))*dt;

%% S^T - F MATRIX FOR GAUSS-LOBATTO
% Calculate (S^T - F) matrix by calling evaluate_wave_rhs on the vectors of
% the canonical base of R^(2n*nk1)
ST_minus_F_gauss_lobatto = sparse(2*kp1*n,2*kp1*n);
for j = 1 : 2*kp1*n
    base_vector = sparse(2*kp1*n, 1);
    base_vector(j) = 1;
    matrix_column = evaluate_wave_rhs(base_vector, c, rho, [0,0], values_gauss_lobatto, derivatives_gauss_lobatto, wg_gauss_lobatto, flux, boundary);
    ST_minus_F_gauss_lobatto(:, j) = matrix_column;
    
end
eigenvalues_gauss_lobatto = eig(full(Minv_gauss_lobatto*ST_minus_F_gauss_lobatto))*dt;

%% PLOTS
% get stability region of RK4 by a Newton iteration to solve R(lambda)=1
t=0:0.0001*pi:2*pi;
z=exp(1i*t);
% to get the Newton method converged, start with the stability region of
% forward Euler, then RK2, then RK3, because each of those is a good
% initial guess for the next higher order RK method

figure(1)
w = z-1;
for i=1:3
    w=w-(1+w+0.5*w.^2-z.^2)./(1+w);
end
hold on
for i=1:4
    w=w-(1+w+w.^2/2+w.^3/6-z.^3)./(1+w+w.^2/2);
end
for i=1:4
    w = w-(1+w+w.^2/2+w.^3/6+w.^4/24-z.^4)./(1+w+w.^2/2+w.^3/6);
end
plot(w,'color','b')
hold on
plot(eigenvalues_gauss, 'or')
title(['Stability Region for RK4 and Gauss Eigenvalues, k=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])


figure(2)
w = z-1;
for i=1:3
    w=w-(1+w+0.5*w.^2-z.^2)./(1+w);
end
hold on
for i=1:4
    w=w-(1+w+w.^2/2+w.^3/6-z.^3)./(1+w+w.^2/2);
end
for i=1:4
    w = w-(1+w+w.^2/2+w.^3/6+w.^4/24-z.^4)./(1+w+w.^2/2+w.^3/6);
end
plot(w,'color','b')
hold on
plot(eigenvalues_gauss_lobatto, 'or')
title(['Stability Region for RK4 and Gauss Lobatto Eigenvalues, k=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])

%% RIGHT HAND SIDE
function rhs = evaluate_wave_rhs(w, c, rho, bc, values, derivatives, weights, flux, boundary)
kp1 = size(values, 1); % degree + 1 
n = length(w)/kp1/2;

v = w(1:kp1*n);
p = w(kp1*n+1:end);

rhs_p = zeros(size(p));
rhs_v = zeros(size(v));

if(flux==0) % LF
    for e=1:n
        p_e = p((e-1)*kp1+1:e*kp1); 
        v_e = v((e-1)*kp1+1:e*kp1);
        
        % interpolate u to quadrature points
        p_quad = (values')*p_e; 
        v_quad = (values')*v_e;
        
        % compute operator at quadrature points and multiply by gradient of
        % test function (S matrix)
        rhs_v((e-1)*kp1+1:e*kp1) = derivatives * (1/rho * weights .* p_quad);
        rhs_p((e-1)*kp1+1:e*kp1) = derivatives * (c^2*rho * weights .* v_quad);
        
        %% LAX-FRIEDRICHS:
        % compute advective numerical flux on the left
        pminus = p_e(1);
        vminus = v_e(1);
        
        if (e==1)
            if (boundary)
                % Dirichlet
                pplus = 2*bc(1)-pminus;
                vplus = vminus;
            else
                % Absorbing condition
                pplus = -pminus - 2*c*rho*vminus; % maybe negative sign
                vplus= vminus;
            end
        else
            pplus = p((e-1)*kp1);
            vplus = v((e-1)*kp1);
        end
        
        numflux_lv = -0.5*(pminus+pplus)/rho + (c/2*(vminus-vplus));
        numflux_lp = -0.5*rho*c^2*(vminus + vplus) + c/2 * (pminus - pplus);
        
        rhs_v((e-1)*kp1+1) = rhs_v((e-1)*kp1+1) - numflux_lv;
        rhs_p((e-1)*kp1+1) = rhs_p((e-1)*kp1+1) - numflux_lp;
        
        % compute advective numerical flux on the right
        pminus = p_e(kp1);
        vminus = v_e(kp1);
        
        if (e==n)
            if (boundary)
                % dirichlet bc
                pplus = 2*bc(2)-pminus;
                vplus = vminus;
            else
                % absorbing condition
                pplus = -pminus+2*c*rho*vminus;
                vplus= vminus;
            end
        else
            pplus = p(e*kp1+1);
            vplus = v(e*kp1+1);
        end
       
        numflux_lv = 0.5*(pminus+pplus)/rho + (c/2*(vminus-vplus));
        numflux_lp = 0.5*rho*c^2*(vminus + vplus) + c/2 * (pminus - pplus);
        
        rhs_p(e*kp1) = rhs_p(e*kp1) - numflux_lp;
        rhs_v(e*kp1) = rhs_v(e*kp1) - numflux_lv;
    end
else   % HDG
    for e=1:n
        p_e = p((e-1)*kp1+1:e*kp1); 
        v_e = v((e-1)*kp1+1:e*kp1);
        
        % interpolate u to quadrature points
        p_quad = (values')*p_e; 
        v_quad = (values')*v_e;
        
        % compute operator at quadrature points and multiply by gradient of
        % test function (S matrix)
        rhs_v((e-1)*kp1+1:e*kp1) = derivatives * (1/rho * weights .* p_quad);
        rhs_p((e-1)*kp1+1:e*kp1) = derivatives * (c^2*rho * weights .* v_quad);
        
        %% HDG:
        % compute advective numerical flux on the left
        pminus = p_e(1);
        vminus = v_e(1);
        
        if (e==1)
            if (boundary)
                % Dirichlet
                numflux_lv = -1/rho*bc(1);
                numflux_lp = -rho*c^2*vminus + 1/c*(pminus - bc(1));
            else
                % Absorbing condition
                numflux_lv = pminus/(2*rho) + c*vminus/2;
                numflux_lp = -rho*c^2/2*vminus + c/2*pminus;
            end
        else
            pplus = p((e-1)*kp1);
            vplus = v((e-1)*kp1);
            numflux_lv = -0.5*(pminus+pplus)/rho + (c/2*(vminus-vplus));
            numflux_lp = -0.5*rho*c^2*(vminus + vplus) + c/2 * (pminus - pplus);
        end
        
        rhs_v((e-1)*kp1+1) = rhs_v((e-1)*kp1+1) - numflux_lv;
        rhs_p((e-1)*kp1+1) = rhs_p((e-1)*kp1+1) - numflux_lp;
        
        % compute advective numerical flux on the right
        pminus = p_e(kp1);
        vminus = v_e(kp1);
        
        if (e==n)
            if (boundary)
                % Dirichlet
                numflux_lv = 1/rho*bc(2);
                numflux_lp = rho*c^2*vminus + 1/c*(pminus - bc(2));
            else
                % Absorbing condition
                numflux_lv = pminus/(2*rho) + c*vminus/2;
                numflux_lp = rho*c^2/2*vminus + c/2*pminus;
            end
        else
            pplus = p(e*kp1+1);
            vplus = v(e*kp1+1);
            numflux_lv = 0.5*(pminus+pplus)/rho + (c/2*(vminus-vplus));
            numflux_lp = 0.5*rho*c^2*(vminus + vplus) + c/2 * (pminus - pplus);            
        end
       
        
        
        rhs_p(e*kp1) = rhs_p(e*kp1) - numflux_lp;
        rhs_v(e*kp1) = rhs_v(e*kp1) - numflux_lv;
    end


end
rhs = [rhs_v; rhs_p];


end

     
