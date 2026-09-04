function [bn, en] = fmwaterfill_gn(gn, b_bar, gap, cb)

    % INPUT
    % gn is the channel gain (a row vector).
    % b_bar is the target bit rate (b_total / Ntot)
    % gap is the gap in dB
    % cb=1 for complex bb and 2 for real bb
    %
    % OUTPUT
    % en is the energy in the nth subchannel
    % bn is the bit in the nth subchannel

    % dB into normal scale
    gap = 10 ^ (gap / 10);

    % Ntot

    [col Ntot] = size(gn);

    if col ~= 1
        error = 1
        return;
    end

    % initialization
    en = zeros(1, Ntot);
    bn = zeros(1, Ntot);

    %%%%%%%%%%%%%%%%%%%%%%%
    % Now do waterfilling %
    %%%%%%%%%%%%%%%%%%%%%%%

    %sort
    [gn_sorted, Index] = sort(gn); % sort gain, and get Index

    gn_sorted = fliplr(gn_sorted); % flip left/right to get the largest
    % gain in leftside
    Index = fliplr(Index); % also flip index

    num_zero_gn = length(find(gn_sorted == 0));
    Nstar = Ntot - num_zero_gn; % number of zero gain subchannels

    % Number of used channels,
    % start from Ntot - (number of zero gain subchannels)

    while (1)
        % The K calculation has been modified in order to
        % accomodate the size of the number
        logK = log(gap) + 1 / Nstar * ((Ntot) * b_bar * 2 * log(2) - sum(log(gn_sorted(1:Nstar))));
        K = exp(logK);
        En_min = K - gap / gn_sorted(Nstar);
        % En_min occurs in the worst channel
        if (En_min < 0)
            Nstar = Nstar - 1; % If negative En, continue with less channels
        else
            break; % If all En positive, done.
        end

    end

    En = K - gap ./ gn_sorted(1:Nstar); % Calculate En
    Bn = (1 / cb) * log2(K * gn_sorted(1:Nstar) / gap); % Calculate bn

    bn(Index(1:Nstar)) = Bn; % return values in original index
    en(Index(1:Nstar)) = En; % return values in original index
