
clear

n = 80;         % number of elements
Tf = 0.2;         % final time
boundary = 1;   % switch between Dirichlet conditions (1) and Absorbing (0)
k = 4;          % polynomial degree
c=1;            % avection term, light velocity
rho=1;          % density
Cr = 1.69/k^2;       % Courant number, MAX IS 1.8, STABLE UNTIL 1.79
flux = 0;       % flux type, 0 = LF, 1 = HDG
nc = k+1;       % number of quadrature points
left = 0;       % left end of the domain
right = 1;      % right end of the domain
plot_accurate = 1; % plot only on nodes (0) or with more resolution (1)

%% ANALYTICAL SOLUTION
% We define two for velocity and pressure
analytical_p = @(x, t) sin(pi * x) * sin(pi * t);
analytical_v = @(x, t) cos(pi * x) * cos(pi * t);

% set quadrature formula and quadrature nodes for integration
% this two Gauss scripts do not neet to be changed
[pg,wg] = get_gauss_quadrature(nc);

% set the node points for the Lagrange polynomials
xunit = get_gauss_lobatto_quadrature(k+1);

%% START THE SIMULATION

% create mesh y and compute mesh size h
% (do not need to be changed)
kp1 = k+1;
y = zeros(2*n,1);
h = zeros(n,1);
for e=1:n
    y(2*e-1) = left + (right-left)*(e-1)/n;
    y(2*e) = left + (right-left)*e/n;
    h(e) = y(2*e)-y(2*e-1);
end

% create the grid of all node points for interpolation
% (do not need to be changed)
for e=1:n
    x((kp1*e-k):kp1*e) = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*xunit);
end

% compute time step from Cr number, adjust to hit the desired final time
% this is correct
dt = Cr * min(h) / c;
NT = round(Tf/dt);
dt = Tf/NT;

disp(['Number of elements: ' num2str(n) ', minimum mesh size: ' ...
    num2str(min(h)) ', time step size: ' num2str(dt) ])

% evaluate reference cell polynomials
[values,derivatives] = evaluate_lagrange_basis(xunit, pg);

%% MASS MATRIX
% create M, we do not create S and F and F flux because it has to be done
% in the rhs part
[values,derivatives] = evaluate_lagrange_basis(xunit, pg);
Me = values * diag(wg) * values'; 
M1inv = sparse(kp1*n,kp1*n);

for e=1:n
    M1inv((kp1*e-k):kp1*e,(kp1*e-k):kp1*e) = inv(0.5*h(e)*Me);
end
Minv = blkdiag(M1inv, M1inv);

%% SET INITIAL CONIDTION
p = analytical_p(x,0); 
v = analytical_v(x,0); 

w = [v, p]'; %column vector

%% RUN TIME LOOP
for m=1:NT
    
    k1 = Minv*evaluate_wave_rhs(w(:,m), c, rho, ...
                 [analytical_p(0,(m-1)*dt); analytical_p(1,(m-1)*dt)],...
                 values, derivatives, wg, flux, boundary);
    k2 = Minv*evaluate_wave_rhs(w(:,m)+0.5*dt*k1, c, rho, ...
                 [analytical_p(0,(m-0.5)*dt); analytical_p(1,(m-0.5)*dt)],...
                 values, derivatives, wg, flux, boundary);
    k3 = Minv*evaluate_wave_rhs(w(:,m)+0.5*dt*k2, c, rho, ...
                 [analytical_p(0,(m-0.5)*dt); analytical_p(1,(m-0.5)*dt)],...
                 values, derivatives, wg, flux, boundary);
    k4 = Minv*evaluate_wave_rhs(w(:,m)+dt*k3, c, rho, ...
                 [analytical_p(0,m*dt); analytical_p(1,m*dt)],...
                 values, derivatives, wg, flux, boundary);
    
    w(:,m+1) = w(:,m) + dt/6*(k1+2*k2+2*k3+k4); 
end

%% PLOTS
% plot the numerical solution (red), the analytical solution (blue), and
% the initial condition

if plot_accurate == 1
    xx_unit = -1:0.05:1;
    xx=zeros(n,length(xx_unit));
    pp_0 = zeros(size(xx));
    pp = zeros(size(xx));
    vv_0 = zeros(size(xx));
    vv = zeros(size(xx));
    val = evaluate_lagrange_basis(xunit, xx_unit);
    for e=1:n
        xx(e,:) = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*xx_unit);
        vv_0(e,:) = val' * w(e*kp1-k:e*kp1,1);
        vv(e,:) = val' * w(e*kp1-k:e*kp1,end);
        pp_0(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,1);
        pp(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,end);
    end
    p_anal = analytical_p(xx, Tf);
    v_anal = analytical_v(xx, Tf);
    % Do the same for velocity

    % plot dummy data to create correct legend despite multiple columns
    figure(1)
    plot(xx(1,:),pp_0(1,:),'k:',xx(1,:),pp(1,:),'r-',xx(1,:),p_anal(1,:),'b');
    hold on
    plot(xx',pp_0','k:');
    plot(xx',pp','r-');
    plot(xx',p_anal','b-');
    hold off
    xlabel('x')
    ylabel('p_h(x)')
    title(['degree=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])
    legend('p_h(x,0)',['p_h(x,' num2str(Tf) ')'],['p(x,' num2str(Tf) ')'])
    figure(2)
    plot(xx(1,:),vv_0(1,:),'k:',xx(1,:),vv(1,:),'r-',xx(1,:),v_anal(1,:),'b');
    hold on
    plot(xx',vv_0','k:');
    plot(xx',vv','r-');
    plot(xx',v_anal','b-');
    hold off
    xlabel('x')
    ylabel('v_h(x)')
    title(['degree=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])
    legend('v_h(x,0)',['v_h(x,' num2str(Tf) ')'],['v(x,' num2str(Tf) ')'])
else
    figure(1)
    plot(x,w(1:n*kp1,1),'k:',x,w(1:n*kp1,end),'r-',x,analytical_v(x,Tf),'b')
    xlabel('x')
    ylabel('v_h(x)')
    title(['degree=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])
    legend('v_h(x,0)',['v_h(x,' num2str(Tf) ')'],['v(x,' num2str(Tf) ')'])
    figure(2)
    plot(x,w(n*kp1+1:end,1),'k:',x,w(n*kp1+1:end,end),'r-',x,analytical_p(x,Tf),'b')
    xlabel('x')
    ylabel('p_h(x)')
    title(['degree=' num2str(k) ', n=' num2str(n) ' elements, dt = ' num2str(dt)])
    legend('p_h(x,0)',['p_h(x,' num2str(Tf) ')'],['p(x,' num2str(Tf) ')'])
end


l2error_p = 0;
linfty_error_p = 0;
[pg_err,wg_err] = get_gauss_quadrature(k+3);
values_err = evaluate_lagrange_basis(xunit, pg_err);
for e=1:n
    sol_num = values_err' * w(kp1*n+(e-1)*kp1+1:kp1*n+e*kp1,end);
    x_err = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*pg_err);
    sol_exact = analytical_p(x_err, Tf);
    l2error_p = l2error_p + h(e)/2 * wg_err' * (sol_num-sol_exact).^2;
    linfty_error_p = max([linfty_error_p; abs(sol_num-sol_exact)]);
end
l2error_p = sqrt(l2error_p);

l2error_v = 0;
linfty_error_v = 0;
[pg_err,wg_err] = get_gauss_quadrature(k+3);
values_err = evaluate_lagrange_basis(xunit, pg_err);
for e=1:n
    sol_num = values_err' * w((e-1)*kp1+1:e*kp1,end);
    x_err = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*pg_err);
    sol_exact = analytical_v(x_err, Tf);
    l2error_v = l2error_v + h(e)/2 * wg_err' * (sol_num-sol_exact).^2;
    linfty_error_v = max([linfty_error_v; abs(sol_num-sol_exact)]);
end
l2error_v = sqrt(l2error_v);

disp(['Velocity error in maximum norm ' num2str(linfty_error_v) ' in L2 norm ' num2str(l2error_v)])
disp(['Pressure error in maximum norm ' num2str(linfty_error_p) ' in L2 norm ' num2str(l2error_p)])

%end

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
                numflux_lp = -rho*c^2*vminus/2 + c*pminus/2;
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
                numflux_lp = rho*c^2*vminus/2 + c*pminus/2;
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
