function rate = nr_mcs(target, rank)
    %NR_MCS Largest PUSCH MCS rate that does not exceed a target rate
    %
    % Spectral efficiencies are those of the 256QAM MCS index table (TS 38.214
    % Table 5.1.3.1-2); one codeword carries every layer, so the rate is the
    % layer count times the per-layer efficiency
    %
    % Args:
    %   target: Target rates in bits per resource element
    %   rank: Number of layers of each transmission
    %
    % Returns:
    %   rate: Quantized rates, zero when even the lowest MCS exceeds the target

    efficiency = [0.2344 0.3770 0.6016 0.8770 1.1758 1.4766 1.6953 1.9141 ...
        2.1602 2.4063 2.5703 2.7305 3.0293 3.3223 3.6094 3.9023 4.2129 ...
        4.5234 4.8164 5.1152 5.3320 5.5547 5.8906 6.2266 6.5703 6.9141 ...
        7.1602 7.4063];
    rate = zeros(size(target));

    for i = 1:numel(target)

        if rank(i) <= 0 || ~isfinite(target(i))
            continue;
        end

        reachable = efficiency(rank(i) * efficiency <= target(i));

        if ~isempty(reachable)
            rate(i) = rank(i) * reachable(end);
        end

    end

end
