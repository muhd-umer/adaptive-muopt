function [J, out] = mmse_info(Y, X, opts)
    %MMSE_INFO Learn the MMSE information matrix of known pilots from received samples
    %
    % The receiver regresses the pilot symbols X on the received samples Y and
    % keeps the residual error covariance E, whose inverse J = inv(E) equals
    % I + A' * Gram * A for the transmit factor A that shaped the pilots. No
    % channel matrix and no noise covariance are formed: the sample covariance
    % of Y whitens colored noise and interference automatically
    %
    % Args:
    %   Y: Ly-by-n received samples
    %   X: L-by-n unit-power pilot symbols that produced Y
    %   opts: Optional ridge (1e-10), errorFloor (1e-10), dof (true divides
    %     the in-sample residual by 1 - Ly/n to undo its optimism), silentRows
    %     (streams the receiver knows were not transmitted), and offsets (each
    %     sample's frequency offset within the band, which lets the LMMSE
    %     receiver vary linearly across it instead of being held constant)
    %
    % Returns:
    %   J: L-by-L information matrix
    %   out: Error covariance E, LMMSE receiver Wle, and sample count

    if nargin < 3 || isempty(opts)
        opts = struct();
    end

    ridge = get_opt(opts, 'ridge', 1e-10);
    errorFloor = get_opt(opts, 'errorFloor', 1e-10);
    dof = get_opt(opts, 'dof', true);
    silentRows = get_opt(opts, 'silentRows', []);
    offsets = get_opt(opts, 'offsets', []);

    if ~isempty(offsets)
        Y = [Y; Y .* reshape(offsets, 1, [])];
    end

    [J, E, Wle] = learn(Y, X, ridge, errorFloor, dof, silentRows);
    out = struct('E', E, 'Wle', Wle, 'samples', size(X, 2));
end

function [J, E, Wle] = learn(Y, X, ridge, errorFloor, dof, silentRows)
    [L, count] = size(X);
    Ly = size(Y, 1);

    if size(Y, 2) ~= count || count < L
        error('mmse_info needs at least as many pilot samples as streams');
    end

    Ryy = hermitian(Y * Y') / count;
    scale = max(real(trace(Ryy)) / max(Ly, 1), 1);
    Wle = (X * Y') / count / (Ryy + ridge * scale * eye(Ly));
    residual = X - Wle * Y;
    E = hermitian(residual * residual') / count;

    if dof && count > Ly + 1
        E = E * count / (count - Ly);
    end

    [V, D] = eig(E);
    e = min(max(real(diag(D)), errorFloor), 1);
    E = hermitian(V * diag(e) * V');
    E(silentRows, :) = 0;
    E(:, silentRows) = 0;
    E(silentRows, silentRows) = eye(numel(silentRows));
    Wle(silentRows, :) = 0;
    J = hermitian(E \ eye(L));
end

function matrix = hermitian(matrix)
    matrix = 0.5 * (matrix + matrix');
end

function value = get_opt(opts, name, default)

    if isfield(opts, name) && ~isempty(opts.(name))
        value = opts.(name);
    else
        value = default;
    end

end
