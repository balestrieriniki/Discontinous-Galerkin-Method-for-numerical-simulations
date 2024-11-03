clear

n = 10;         % number of elements
Tf = 0.003;       % final time
boundary = 1;   % switch between Dirichlet conditions (1) and Absorbing (0)
k = 10;          % polynomial degree
c=340;            % avection term, light velocity
rho=1.2;          % density
Cr = 0.4;       % Courant number -> sets time step size in dt = Cr * h / a
flux = 1;       % flux type, 0 = LF, 1 = HDG
nc = k+1;       % number of quadrature points
left = 0;       % left end of the domain
right = 1;      % right end of the domain
plot_accurate = 1; % plot only on nodes (0) or with more resolution (1)

%% ANALYTICAL SOLUTION
% We define two for velocity and pressure
analytical_p = @(x) exp(-(x-0.5).^2/(0.02)^2);
analytical_v = @(x) zeros(size(x));

% set quadrature formula and quadrature nodes for integration
[pg,wg] = get_gauss_quadrature(nc);

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

% create the grid of all node points for interpolation
for e=1:n
    x((kp1*e-k):kp1*e) = y(2*e-1)+(y(2*e)-y(2*e-1))*(0.5+0.5*xunit);
end

% compute time step from Cr number, adjust to hit the desired final time
dt = Cr * min(h) / (c * (k^1.5));
%dt = 6e-4;
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
p = analytical_p(x); 
v = analytical_v(x); 

w = [v, p]'; %column vector

%% RUN TIME LOOP
for m=1:NT
    
    k1 = Minv*evaluate_wave_rhs(w(:,m), c, rho, ...
                 [analytical_p(0); analytical_p(1)],...
                 values, derivatives, wg, flux, boundary);
    k2 = Minv*evaluate_wave_rhs(w(:,m)+0.5*dt*k1, c, rho, ...
                 [analytical_p(0); analytical_p(1)],...
                 values, derivatives, wg, flux, boundary);
    k3 = Minv*evaluate_wave_rhs(w(:,m)+0.5*dt*k2, c, rho, ...
                 [analytical_p(0); analytical_p(1)],...
                 values, derivatives, wg, flux, boundary);
    k4 = Minv*evaluate_wave_rhs(w(:,m)+dt*k3, c, rho, ...
                 [analytical_p(0); analytical_p(1)],...
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
        vv_1(e,:) = val' * w(e*kp1-k:e*kp1,161);
        vv_2(e,:) = val' * w(e*kp1-k:e*kp1,322);
        vv_3(e,:) = val' * w(e*kp1-k:e*kp1,483);
        vv_4(e,:) = val' * w(e*kp1-k:e*kp1,645);
        vv(e,:) = val' * w(e*kp1-k:e*kp1,end);
        pp_0(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,1);
        pp_1(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,161);
        pp_2(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,322);
        pp_3(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,483);
        pp_4(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,645);
        pp(e,:) = val' * w(kp1*n+e*kp1-k:kp1*n+e*kp1,end);
    end
   

    % plot dummy data to create correct legend despite multiple columns
    figure(1)
    plot(xx(1,:),pp_0(1,:),'-k',xx(1,:),pp_1(1,:),'-m', xx(1,:),pp_2(1,:),'-c', xx(1,:),pp_3(1,:),'-b', xx(1,:),pp_4(1,:),'-g', xx(1,:),pp(1,:),'-r');
    hold on
    plot(xx',pp_0','-k', 'Linewidth', 2);
    plot(xx',pp_1','-m', 'Linewidth', 2);
    plot(xx',pp_2','-c', 'Linewidth', 2);
    plot(xx',pp_3','-b', 'Linewidth', 2);
    plot(xx',pp_4','-g', 'Linewidth', 2);
    plot(xx',pp','-r', 'Linewidth', 2);
    
    hold off
    xlabel('Domain (x)')
    ylabel('Pressure (p)')
    title(['Pressure change over time for dt=0.0006 s'])
    legend('p = 0.000 s', 'p = 0.0006 s', 'p = 0.0012 s', 'p = 0.0018 s', 'p = 0.0024 s', 'p = 0.003 s')
    
    
figure(2)

subplot(2,3,1)
plot(xx', pp_0', '-r');
xlabel('Domain (x)')
ylabel('Pressure (p)')

subplot(2,3,2)
plot(xx', pp_1', '-r');


subplot(2,3,3)
plot(xx', pp_2', '-r');


subplot(2,3,4)
plot(xx', pp_3', '-r');
xlabel('Domain (x)')
ylabel('Pressure (p)')

subplot(2,3,5)
plot(xx', pp_4', '-r');


subplot(2,3,6)
plot(xx', pp', '-r');


sgtitle('Pressure change over time for dt=0.0006 s')

    
    
    figure(3)
    plot(xx(1,:),vv_0(1,:),'-k',xx(1,:),vv_1(1,:),'-m', xx(1,:),vv_2(1,:),'-c', xx(1,:),vv_3(1,:),'-b', xx(1,:),vv_4(1,:),'-g', xx(1,:),vv(1,:),'-r');
    hold on
    plot(xx',vv_0','-k', 'Linewidth', 2);
    plot(xx',vv_1','-m', 'Linewidth', 2);
    plot(xx',vv_2','-c', 'Linewidth', 2);
    plot(xx',vv_3','-b', 'Linewidth', 2);
    plot(xx',vv_4','-g', 'Linewidth', 2);
    plot(xx',vv','-r', 'Linewidth', 2);
    
    hold off
    xlabel('Domain (x)')
    ylabel('Velocity (v)')
    title(['Velocity change over time for dt=0.0006 s'])
    legend('v = 0.000 s', 'v = 0.0006 s', 'v = 0.0012 s', 'v = 0.0018 s', 'v = 0.0024 s', 'v = 0.003 s')
    

    figure(4)
    subplot(2,3,1)
    plot(xx',vv_0','-r');
    xlabel('Domain (x)')
ylabel('Velocity (v)')

    subplot(2,3,2)
    plot(xx',vv_1','-r');

    subplot(2,3,3)
    plot(xx',vv_2','-r');

    subplot(2,3,4)
    plot(xx',vv_3','-r');
    xlabel('Domain (x)')
ylabel('Velocity (v)')

    subplot(2,3,5)
    plot(xx',vv_4','-r');
    subplot(2,3,6)
    plot(xx',vv','-r');
    sgtitle('Velocity change over time for dt=0.0006 s')
    
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
    disp("you can't use LF")
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
                % Absorbing 
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
                % Absorbing 
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


