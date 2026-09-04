function [bn, en] = fmwaterfill_gn(gn, b_bar, gap, cb)
    %FMWATERFILL_GN Water-filling allocation for a frequency-selective channel
    %   [bn, en] = FMWATERFILL_GN(gn, b_bar, gap, cb) distributes rate and energy
    %   across subchannels using the classic water-filling solution while honoring
    %   a gap parameter and complex/real baseband scaling.
    %
    %   Inputs:
    %       gn      row vector of channel gains.
    %       b_bar   target bit rate per tone (b_total / Ntot).
    %       gap     gap in dB.
    %       cb      equals 1 for complex baseband and 2 for real baseband.
    %
    %   Outputs:
    %       en      energy allocation per subchannel.
    %       bn      bit allocation per subchannel.

    % convert dB gap to linear scale
    %
    %   Author: Muhammad Umer
    %   Organization: Stanford University
    gap = 10 ^ (gap / 10);

    % capture number of tones
    [rows, Ntot] = size(gn);

    if rows ~= 1
        error('gn must be a row vector (1 x N)');
    end

    % initialization
    en = zeros(1, Ntot);
    bn = zeros(1, Ntot);

    %%%%%%%%%%%%%%%%%%%%%%%
    % now do water-filling %
    %%%%%%%%%%%%%%%%%%%%%%%

    % sort gains and keep indices
    [gn_sorted, Index] = sort(gn); % sort gain and get index

    gn_sorted = fliplr(gn_sorted); % flip left/right to get the largest gain leftmost
    Index = fliplr(Index); % also flip index

    num_zero_gn = length(find(gn_sorted == 0));
    Nstar = Ntot - num_zero_gn; % number of zero-gain subchannels

    % number of used channels, start from Ntot minus zero-gain subchannels
    while (1)
        % the K calculation has been modified in order to accommodate scaling
        logK = log(gap) + 1 / Nstar * ((Ntot) * b_bar * 2 * log(2) - sum(log(gn_sorted(1:Nstar))));
        K = exp(logK);
        En_min = K - gap / gn_sorted(Nstar);
        % en_min occurs in the worst channel
        if (En_min < 0)
            Nstar = Nstar - 1; % if negative En, continue with fewer channels
        else
            break; % if all En positive, done
        end

    end

    En = K - gap ./ gn_sorted(1:Nstar); % calculate En
    Bn = (1 / cb) * log2(K * gn_sorted(1:Nstar) / gap); % calculate bn

    bn(Index(1:Nstar)) = Bn; % return values in original index
    en(Index(1:Nstar)) = En; % return values in original index
end
