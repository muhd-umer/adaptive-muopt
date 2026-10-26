function ctrl = ctrl_init(U, link)
    %CTRL_INIT Empty control link between the gNB scheduler and its UEs
    %
    % Args:
    %   U: Number of UEs
    %   link: Delivery model with delay (slots from decision to activation),
    %     pDrop (probability a message is lost), pLate (probability it lands
    %     one slot late), and lateSlots (extra delay of a late message)
    %
    % Returns:
    %   ctrl: Message queue and per-UE accounting

    ctrl = struct();
    ctrl.link = link;
    ctrl.queue = struct('ue', {}, 'sent', {}, 'due', {}, 'kind', {}, ...
        'bits', {}, 'payload', {}, 'dropped', {});
    ctrl.messages = zeros(U, 1);
    ctrl.bits = zeros(U, 1);
    ctrl.dropped = zeros(U, 1);
    ctrl.late = zeros(U, 1);
    ctrl.delivered = zeros(U, 1);
end
