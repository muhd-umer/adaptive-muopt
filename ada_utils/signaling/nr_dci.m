function dci = nr_dci(catalogs, Lxu, NRB, dmrsType)
    %NR_DCI Bit-exact transmit-action fields of DCI format 0_1 and their codec
    %
    % Codebook PUSCH carries "Precoding information and number of layers"
    % (TS 38.212 Tables 7.3.1.1.2-2 to -5), the 2-bit TPC command, and an SRI
    % that is empty for a single SRS resource. Non-codebook PUSCH has no
    % precoding field; its SRI selects the layers among one single-port SRS
    % resource per UE port (TS 38.212 Table 7.3.1.1.2-28 to -31). The other
    % DCI 0_1 fields are counted for control-channel load, with widths for a
    % single-carrier, single-BWP, dynamic-grant configuration
    %
    % Args:
    %   catalogs: 1-by-U cell of nr_catalog outputs
    %   Lxu: Antenna ports per UE
    %   NRB: Carrier bandwidth in resource blocks
    %   dmrsType: DMRS configuration type, which sets the antenna-ports width
    %
    % Returns:
    %   dci: Per-UE field widths, the accumulated TPC steps in dB, full DCI 0_1
    %     sizes of both PUSCH schemes, and handles bits = dci.encode(u, row,
    %     tpc) and msg = dci.decode(u, bits) for the codebook fields

    U = numel(catalogs);
    antennaPorts = 3 + dmrsType;
    common = 1 + ceil(log2(NRB * (NRB + 1) / 2)) + 4 + 5 + 1 + 2 + 4 + 1 + antennaPorts + 2 + ...
        1 + 1;
    dci = struct();
    dci.precoding = cellfun(@(c) c.bits, catalogs);
    dci.tpc = 2 * ones(1, U);
    dci.sri = zeros(1, U);
    dci.sriNonCodebook = arrayfun(@(L) ceil(log2(sum(arrayfun(@(k) nchoosek(L, k), 1:L)))), ...
        reshape(Lxu, 1, []));
    dci.design = dci.precoding + dci.tpc + dci.sri;
    dci.crc = 24;
    dci.dci01 = common + dci.design + dci.crc;
    dci.dci01NonCodebook = common + dci.tpc + dci.sriNonCodebook + dci.crc;
    dci.tpcStepsDb = [-1 0 1 3];
    dci.encode = @(u, row, tpc) encode(dci, catalogs{u}, u, row, tpc);
    dci.decode = @(u, bits) decode(dci, catalogs{u}, u, bits);
end

function bits = encode(dci, catalog, u, row, tpc)

    if row < 1 || row > numel(catalog.rows) || row ~= fix(row) || ~ismember(tpc, 0:3)
        error('invalid precoding or TPC code point');
    end

    bits = [to_bits(row - 1, dci.precoding(u)), to_bits(tpc, dci.tpc(u)), ...
        false(1, dci.sri(u))];
end

function msg = decode(dci, catalog, u, bits)

    if numel(bits) ~= dci.design(u)
        error('grant carries %d bits but the configured fields need %d', numel(bits), ...
            dci.design(u));
    end

    code = from_bits(bits(1:dci.precoding(u)));
    tpc = from_bits(bits(dci.precoding(u) + (1:dci.tpc(u))));
    msg = struct('row', code + 1, 'valid', code < numel(catalog.rows), 'rank', 0, ...
        'tpmi', -1, 'tpc', tpc, 'tpcDb', dci.tpcStepsDb(tpc + 1));

    if msg.valid
        msg.rank = catalog.rows(msg.row).rank;
        msg.tpmi = catalog.rows(msg.row).tpmi;
    end

end

function bits = to_bits(value, width)
    bits = logical(bitget(value, width:-1:1));
end

function value = from_bits(bits)
    value = sum(double(bits) .* 2 .^ (numel(bits) - 1:-1:0));
end
