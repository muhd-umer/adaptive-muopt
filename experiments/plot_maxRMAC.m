%PLOT_MAXRMAC Result figures of the NR maxRMAC experiment
%
% Reads experiments/results/nrPUSCH_maxRMAC.mat and writes one PDF per figure
% to experiments/results/maxrmac/. Colors follow the method class (genie
% reference, NR-native, compressed covariance, reciprocity) and every mark
% is named directly, so no figure relies on color alone. Time-series figures
% are drawn only when the saved statistics carry per-slot traces

clear;
clc;
root = fileparts(fileparts(mfilename('fullpath')));
source = fullfile(root, 'experiments', 'results', 'nrPUSCH_maxRMAC.mat');
outputDir = fullfile(root, 'experiments', 'results', 'maxrmac');

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

saved = load(source);
style = figure_style();

plot_weighted_rate(saved.stats, style, fullfile(outputDir, 'weighted_rate.pdf'));
plot_gap(saved.stats, style, fullfile(outputDir, 'gap_to_canonical.pdf'));
plot_rate_feedback(saved.stats, style, fullfile(outputDir, 'rate_vs_feedback.pdf'));
plot_throughput(saved.stats, style, fullfile(outputDir, 'throughput.pdf'));
plot_signaling(saved.stats, style, fullfile(outputDir, 'signaling.pdf'));

if isfield(saved.stats, 'trace')
    slot = 1e-3 / (saved.cfg.carrier.scs / 15);
    plot_traces(saved.stats, saved.cfg, slot, style, outputDir);
else
    fprintf('no per-slot traces saved; rerun the experiment for the time-series figures\n');
end

fprintf('figures written to %s\n', outputDir);

function style = figure_style()
    %FIGURE_STYLE Fonts, ink, and the class palette shared by every figure
    style = struct();
    style.font = 'Source Serif 4';
    style.size = 9;
    style.ink = [0.043 0.043 0.043];
    style.secondary = [0.322 0.318 0.306];
    style.grid = [0.88 0.88 0.87];
    style.surface = [1 1 1];
    style.genie = [0.62 0.62 0.60];
    style.classes = struct('genie', style.genie, 'nr', hex('#2a78d6'), ...
        'covariance', hex('#eb6834'), 'reciprocity', hex('#1baf7a'));
    style.ramp = [hex('#9cc3f2'); hex('#5a9ae6'); hex('#1f5fae')];
    style.labels = containers.Map( ...
        {'canonical', 'nr-oracle', 'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', ...
        'ue-pilot-free', 'hold'}, ...
        {'Canonical (genie)', 'NR oracle (genie)', 'NR, SRS estimate', ...
        'NR, pilot-learned', 'Covariance feedback', 'Reciprocity, flat PSD', ...
        'Reciprocity, free PSD', 'Fixed precoder (genie)'});
    style.class = containers.Map( ...
        {'canonical', 'nr-oracle', 'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', ...
        'ue-pilot-free', 'hold'}, ...
        {'genie', 'genie', 'nr', 'nr', 'covariance', 'reciprocity', 'reciprocity', 'genie'});
end

function color = hex(code)
    color = sscanf(code(2:end), '%2x%2x%2x', [1 3]) / 255;
end

function [fig, layout] = new_figure(style, width, height, rows, columns)
    fig = figure('Units', 'inches', 'Position', [1 1 width height], 'Color', style.surface, ...
        'Visible', 'off');
    layout = tiledlayout(fig, rows, columns, 'TileSpacing', 'compact', 'Padding', 'compact');
end

function ax = styled_axes(layout, style)
    ax = nexttile(layout);
    hold(ax, 'on');
    set(ax, 'FontName', style.font, 'FontSize', style.size, 'XColor', style.secondary, ...
        'YColor', style.secondary, 'Color', style.surface, 'Box', 'off', ...
        'GridColor', style.grid, 'GridAlpha', 1, 'TickDir', 'out', 'LineWidth', 0.6);
end

function finish(fig, style, file)
    set(findall(fig, '-property', 'FontName'), 'FontName', style.font);
    exportgraphics(fig, file, 'ContentType', 'vector', 'BackgroundColor', style.surface);
    close(fig);
end

function colors = class_colors(names, style)
    colors = zeros(numel(names), 3);

    for m = 1:numel(names)
        colors(m, :) = style.classes.(style.class(names{m}));
    end

end

function labels = method_labels(names, style)
    labels = cellfun(@(name) style.labels(name), names, 'UniformOutput', false);
end

function plot_weighted_rate(stats, style, file)
    %PLOT_WEIGHTED_RATE Weighted information rate of every method with 95% intervals
    %
    % The methods differ by a few percent, so the rates are drawn as dots on
    % a zoomed axis; bars would misstate the magnitude on an axis not at zero
    [fig, layout] = new_figure(style, 3.4, 2.9, 1, 1);

    [value, order] = sort(stats.objective, 'ascend');
    names = stats.names(order);
    ci = stats.objectiveCI(order);
    colors = class_colors(names, style);
    ax = styled_axes(layout, style);

    for m = 1:numel(value)
        plot(ax, [value(m) - ci(m), value(m) + ci(m)], [m, m], '-', 'Color', style.ink, ...
            'LineWidth', 1);
        plot(ax, value(m), m, 'o', 'MarkerSize', 8, 'MarkerFaceColor', colors(m, :), ...
            'MarkerEdgeColor', style.surface, 'LineWidth', 1.5);
        text(ax, value(m) + ci(m) + 0.06, m, sprintf('%.2f', value(m)), ...
            'FontName', style.font, 'FontSize', style.size - 1, 'Color', style.ink, ...
            'VerticalAlignment', 'middle');
    end

    set(ax, 'YTick', 1:numel(value), 'YTickLabel', method_labels(names, style), ...
        'TickLabelInterpreter', 'none', 'XGrid', 'on', 'YGrid', 'on');
    ylim(ax, [0.4, numel(value) + 0.6]);
    xlim(ax, [floor(2 * (min(value) - 0.3)) / 2, max(value) + 0.5]);
    xlabel(ax, 'Weighted rate (bit/RE)', 'Color', style.ink);

    finish(fig, style, file);
end

function plot_gap(stats, style, file)
    %PLOT_GAP Paired weighted-rate gap of every method to the canonical optimum
    [fig, layout] = new_figure(style, 4.4, 3.1, 1, 1);

    keep = ~strcmp(stats.names, 'canonical');
    names = stats.names(keep);
    gap = 100 * stats.gaps.canonical.mean(keep);
    ci = 100 * stats.gaps.canonical.ci(keep);
    [gap, order] = sort(gap, 'descend');
    names = names(order);
    ci = ci(order);
    ax = styled_axes(layout, style);
    bars = barh(ax, gap, 0.62, 'FaceColor', 'flat', 'EdgeColor', style.surface, ...
        'LineWidth', 1.5);
    bars.CData = class_colors(names, style);
    errorbar(ax, gap, 1:numel(gap), ci, 'horizontal', 'LineStyle', 'none', ...
        'Color', style.ink, 'LineWidth', 0.8, 'CapSize', 3);

    for m = 1:numel(gap)
        text(ax, gap(m) + ci(m) + 0.4, m, sprintf('%.1f%%', gap(m)), 'FontName', style.font, ...
            'FontSize', style.size - 1, 'Color', style.ink, 'VerticalAlignment', 'middle');
    end

    set(ax, 'YTick', 1:numel(gap), 'YTickLabel', method_labels(names, style), ...
        'TickLabelInterpreter', 'none', 'XGrid', 'on');
    xlim(ax, [0, max(gap + ci) + 4]);
    xlabel(ax, 'Gap to canonical optimum (%)', 'Color', style.ink);

    finish(fig, style, file);
end

function plot_rate_feedback(stats, style, file)
    %PLOT_RATE_FEEDBACK Gap to canonical against the design signaling it costs
    [fig, layout] = new_figure(style, 3.4, 2.9, 1, 1);
    ax = styled_axes(layout, style);
    learners = {'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', 'ue-pilot-free'};
    markers = containers.Map({'nr', 'covariance', 'reciprocity'}, {'o', 's', '^'});
    oracle = 100 * stats.gaps.canonical.mean(strcmp(stats.names, 'nr-oracle'));
    yline(ax, oracle, '--', 'Color', style.genie, 'LineWidth', 1);
    text(ax, 1.2e2, oracle + 0.55, 'NR oracle, perfect CSI', 'FontName', style.font, ...
        'FontSize', style.size - 1, 'Color', style.secondary);
    offsets = containers.Map(learners, {[1.2, 0.7], [1.2, -0.55], [0.87, 0.55], ...
        [1.2, 0.45], [1.2, -0.45]});
    alignment = containers.Map(learners, {'left', 'left', 'right', 'left', 'left'});

    for i = 1:numel(learners)
        m = find(strcmp(stats.names, learners{i}), 1);
        x = max(stats.designRate(m), 1);
        y = 100 * stats.gaps.canonical.mean(m);
        ci = 100 * stats.gaps.canonical.ci(m);
        group = style.class(learners{i});
        errorbar(ax, x, y, ci, 'LineStyle', 'none', 'Color', style.ink, 'LineWidth', 0.8, ...
            'CapSize', 3);
        plot(ax, x, y, markers(group), 'MarkerSize', 8, 'MarkerFaceColor', ...
            style.classes.(group), 'MarkerEdgeColor', style.surface, 'LineWidth', 1.5);
        shift = offsets(learners{i});
        text(ax, x * shift(1), y + shift(2), style.labels(learners{i}), 'FontName', style.font, ...
            'FontSize', style.size - 1, 'Color', style.ink, ...
            'HorizontalAlignment', alignment(learners{i}));
    end

    set(ax, 'XScale', 'log', 'XGrid', 'on', 'YGrid', 'on');
    xlim(ax, [1e2, 2e6]);
    ylim(ax, [0, 12]);
    xlabel(ax, 'Design signaling (bit/s)', 'Color', style.ink);
    ylabel(ax, 'Gap to canonical optimum (%)', 'Color', style.ink);
    finish(fig, style, file);
end

function plot_throughput(stats, style, file)
    %PLOT_THROUGHPUT Information rate against realistic link throughput per learner
    learners = {'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', 'ue-pilot-free'};
    rows = cellfun(@(name) find(strcmp(stats.names, name), 1), learners);
    values = [stats.objective(rows), stats.throughput(rows), stats.netThroughput(rows)];
    errors = [stats.objectiveCI(rows), stats.throughputCI(rows), stats.netThroughputCI(rows)];
    series = {'Weighted rate', 'Link throughput', 'Net of reference signals'};
    [fig, layout] = new_figure(style, 7, 3, 1, 1);
    ax = styled_axes(layout, style);
    bars = bar(ax, values, 0.82, 'EdgeColor', style.surface, 'LineWidth', 1.5);

    for k = 1:3
        bars(k).FaceColor = style.ramp(k, :);
        x = bars(k).XEndPoints;
        errorbar(ax, x, values(:, k), errors(:, k), 'LineStyle', 'none', 'Color', style.ink, ...
            'LineWidth', 0.8, 'CapSize', 2, 'HandleVisibility', 'off');
    end

    set(ax, 'XTick', 1:numel(learners), 'XTickLabel', method_labels(learners, style), ...
        'TickLabelInterpreter', 'none', 'YGrid', 'on');
    ylabel(ax, 'bit/RE', 'Color', style.ink);
    legend(ax, series, 'Location', 'northoutside', 'Orientation', 'horizontal', ...
        'Box', 'off', 'TextColor', style.ink, 'FontName', style.font);
    finish(fig, style, file);
end

function plot_signaling(stats, style, file)
    %PLOT_SIGNALING Design bits and full DCI load per learner
    learners = {'nr-csi', 'nr-pilot', 'cov-pilot', 'ue-pilot', 'ue-pilot-free'};
    rows = cellfun(@(name) find(strcmp(stats.names, name), 1), learners);
    measures = {stats.designRate(rows), stats.controlRate(rows)};
    labels = {'Design signaling (bit/s)', 'DCI payload (bit/s)'};
    [fig, layout] = new_figure(style, 6, 2.9, 1, 2);
    colors = class_colors(learners, style);

    for k = 1:2
        ax = styled_axes(layout, style);
        value = measures{k}(:);
        shown = max(value, 1);
        bars = barh(ax, shown, 0.62, 'FaceColor', 'flat', 'EdgeColor', style.surface, ...
            'LineWidth', 1.5);
        bars.CData = colors;

        for m = 1:numel(value)
            text(ax, shown(m) * 1.3, m, compact(value(m)), 'FontName', style.font, ...
                'FontSize', style.size - 1, 'Color', style.ink, 'VerticalAlignment', 'middle');
        end

        set(ax, 'XScale', 'log', 'XGrid', 'on', 'YTick', 1:numel(learners), ...
            'XTick', 10 .^ (0:2:6));
        xlim(ax, [1, 1e7]);

        if k == 1
            set(ax, 'YTickLabel', method_labels(learners, style), 'TickLabelInterpreter', 'none');
        else
            set(ax, 'YTickLabel', {});
        end

        xlabel(ax, labels{k}, 'Color', style.ink);
    end

    finish(fig, style, file);
end

function text = compact(value)

    if value == 0
        text = '0';
    elseif value >= 1e6
        text = sprintf('%.2gM', value / 1e6);
    elseif value >= 1e3
        text = sprintf('%.3gk', value / 1e3);
    else
        text = sprintf('%.0f', value);
    end

end

function plot_traces(stats, cfg, slot, style, outputDir)
    %PLOT_TRACES Weighted rate and cumulative design signaling over time
    shown = {'canonical', 'nr-oracle', 'nr-pilot', 'cov-pilot', 'ue-pilot'};
    rows = cellfun(@(name) find(strcmp(stats.names, name), 1), shown);
    time = stats.trace.slots * slot * 1e3;
    dashes = containers.Map(shown, {'-', '--', '-', '-', '-'});
    tones = containers.Map(shown, {style.ink, style.genie, style.classes.nr, ...
        style.classes.covariance, style.classes.reciprocity});
    panels = {'objective', 'designBits'};
    names = {'rate_over_time', 'feedback_over_time'};
    labels = {'Weighted rate (bit/RE)', 'Cumulative design signaling (kbit)'};

    for p = 1:2
        [fig, layout] = new_figure(style, 5.2, 3.1, 1, 1);
        ax = styled_axes(layout, style);

        for i = 1:numel(shown)
            m = rows(i);
            value = stats.trace.(panels{p})(m, :);
            spread = stats.trace.([panels{p}, 'CI'])(m, :);

            if p == 2

                if ismember(shown{i}, {'canonical', 'nr-oracle'})
                    continue;
                end

                value = cumsum(value) / 1e3;
                spread = zeros(size(value));
            end

            fill(ax, [time, fliplr(time)], [value - spread, fliplr(value + spread)], ...
                tones(shown{i}), 'FaceAlpha', 0.14, 'EdgeColor', 'none', 'HandleVisibility', 'off');
            plot(ax, time, value, dashes(shown{i}), 'Color', tones(shown{i}), 'LineWidth', 2, ...
                'DisplayName', style.labels(shown{i}));
        end

        xline(ax, cfg.warmupSlots * slot * 1e3, ':', 'Color', style.secondary, ...
            'HandleVisibility', 'off');
        set(ax, 'XGrid', 'on', 'YGrid', 'on');
        xlabel(ax, 'Time (ms)', 'Color', style.ink);
        ylabel(ax, labels{p}, 'Color', style.ink);

        if p == 2
            set(ax, 'YScale', 'log');
        end

        legend(ax, 'Location', 'eastoutside', 'Box', 'off', 'TextColor', style.ink, ...
            'FontName', style.font);
        finish(fig, style, fullfile(outputDir, [names{p}, '.pdf']));
    end

end
