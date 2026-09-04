% function [bn, en, Nstar] = waterfill_gn(gn, E_bar, gap, cb)
%
% This function allows the energy budget to be divided by the total number
% of dimensions (unlike waterfill, for which only the total energy is
% specified).  THIS FUNCTION ACCEPTS ENERGY/INPUT DIMENSION NOT TOTAL
% ENERGY.
%
% So, for instance, if there are (N+ nu) temporal dimensions
% with only N carrying energy, the normalization in construction E_bar
% would divide by N+nu.  Instead if spatial with Lx dimensions, E_bar would
% divide by Lx.  Basically, the program user of waterfill_gn decides this
% normalization.
%
% INPUT
% gn is the channel gain (a row vector).
% E_bar is the normalized power constraint (E_total / Ntot)
% gap is the gap in dB
% cb = 1 for complex bb and cb=2 for real bb (if not given, assumes cb=2)
%
% OUTPUT
% en is the energy in the nth subchannel; for complex baseband divide by 2
% bn is the bit in the nth subchannel; for complex baseband multiply by 2
% Nstar is the number of energized dimensions
%
% dB into normal scale
function [bn , en , Nstar] = waterfill_gn(gn, E_bar, gap , cb)
gap = 10^(gap/10);

if nargin < 4
    cb=2;
end

[col, Ntot] = size(gn);

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
[gn_sorted, Index] = sort(gn);   % sort gain, and get Index

gn_sorted = fliplr(gn_sorted);   % flip left/right to get the largest 
                                 % gain in leftside
Index = fliplr(Index);           % also flip index  

num_zero_gn = length(find(gn_sorted == 0)); 
Nstar = Ntot - num_zero_gn;      % number of zero gain subchannels

 	                             % Number of used channels, 
 	                             % start from Ntot - (number of zero gain subchannels)

while(1) 
                                 % The K calculation has been modified in order to 
                                 % accomodate the size of the number
    K = 1/Nstar * (Ntot * E_bar + gap * sum(1 ./ gn_sorted(1:Nstar)));
 	En_min = K - gap/gn_sorted(Nstar);	
                                 % En_min occurs in the worst channel
 	if (En_min<0)		
    		Nstar = Nstar - 1;   % If negative En, continue with less channels
 	else 
    		break;               % If all En positive, done.
 	end
end

En = K - gap./gn_sorted(1:Nstar); % Calculate En
Bn =(1/cb) * log2(K * gn_sorted(1:Nstar)/gap); 	% Calculate bn

bn(Index(1:Nstar))=Bn;		% return values in original index
en(Index(1:Nstar))=En;		% return values in original index