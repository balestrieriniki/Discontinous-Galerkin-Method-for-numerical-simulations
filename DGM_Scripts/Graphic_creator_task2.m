% Define the vectors for n and k
N_vector = [5 10 20 40 80];
k_vector = [1 2 3 4];

% Preallocate error matrices
errors_v_LF = zeros(length(N_vector), length(k_vector));
errors_p_LF = zeros(length(N_vector), length(k_vector));
errors_v_HDG = zeros(length(N_vector), length(k_vector));
errors_p_HDG = zeros(length(N_vector), length(k_vector));

% Compute errors for Lax-Friedrichs method
for i = 1 : length(N_vector)
    for j = 1 : length(k_vector)
        [~, error] = acustic_wave_solver(N_vector(i), k_vector(j), 0);
        errors_v_LF(i, j) = error(1);
        errors_p_LF(i, j) = error(2);
    end
end

% Compute errors for Hybrid Discontinuous Galerkin method
for i = 1 : length(N_vector)
    for j = 1 : length(k_vector)
        [~, error] = acustic_wave_solver(N_vector(i), k_vector(j), 1);
        errors_v_HDG(i, j) = error(1);
        errors_p_HDG(i, j) = error(2);
    end
end

% Create table for Lax-Friedrichs method
LF_Table_Velocity = array2table(errors_v_LF, 'VariableNames', {'k=1', 'k=2', 'k=3', 'k=4'}, 'RowNames', {'n=5', 'n=10', 'n=20', 'n=40', 'n=80'});
LF_Table_Pressure = array2table(errors_p_LF, 'VariableNames', {'k=1', 'k=2', 'k=3', 'k=4'}, 'RowNames', {'n=5', 'n=10', 'n=20', 'n=40', 'n=80'});

% Create table for Hybrid Discontinuous Galerkin method
HDG_Table_Velocity = array2table(errors_v_HDG, 'VariableNames', {'k=1', 'k=2', 'k=3', 'k=4'}, 'RowNames', {'n=5', 'n=10', 'n=20', 'n=40', 'n=80'});
HDG_Table_Pressure = array2table(errors_p_HDG, 'VariableNames', {'k=1', 'k=2', 'k=3', 'k=4'}, 'RowNames', {'n=5', 'n=10', 'n=20', 'n=40', 'n=80'});

% Display tables
disp('Lax-Friedrichs Velocity Errors:');
disp(LF_Table_Velocity);
disp('Lax-Friedrichs Pressure Errors:');
disp(LF_Table_Pressure);

disp('HDG Velocity Errors:');
disp(HDG_Table_Velocity);
disp('HDG Pressure Errors:');
disp(HDG_Table_Pressure);
