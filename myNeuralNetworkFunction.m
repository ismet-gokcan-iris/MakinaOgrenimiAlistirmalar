function [Y,Xf,Af] = myNeuralNetworkFunction(X,~,~)
%MYNEURALNETWORKFUNCTION neural network simulation function.
%
% Auto-generated by MATLAB, 03-Jul-2020 15:35:01.
%
% [Y] = myNeuralNetworkFunction(X,~,~) takes these arguments:
%
%   X = 0xTS cell, 0 inputs over TS timesteps
%
% and returns:
%   Y = 0xTS cell of 0 outputs over TS timesteps.
%
% where Q is number of samples (or series) and TS is the number of timesteps.

%#ok<*RPMT0>

% ===== NEURAL NETWORK CONSTANTS =====


% ===== SIMULATION ========

% Format Input Arguments
isCellX = iscell(X);
if iscell(X)
    X = {X};
end

% Dimensions
TS = size(X,2); % timesteps

% Allocate Outputs
Y = cell(0,TS);

% Time loop
for ts=1:TS
    
end

% Final Delay States
Xf = cell(0,0);
Af = cell(0,0);

% Format Output Arguments
if ~isCellX
    Y = cell2mat(Y);
end
end

% ===== MODULE FUNCTIONS ========

