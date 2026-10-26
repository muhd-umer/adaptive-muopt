function fraction = data_fraction(env, srsInSlot)
    %DATA_FRACTION Share of an uplink slot's symbols left for PUSCH data
    %
    % Args:
    %   env: Environment from nr_setup
    %   srsInSlot: Whether the SRS occupies symbols of this uplink slot
    %
    % Returns:
    %   fraction: Data symbols over all symbols of the slot

    lost = env.dmrsSymbols + env.srsSymbols * srsInSlot;
    fraction = (env.numSymbols - lost) / env.numSymbols;
end
