function [ dt mag_sup corr_sup ] = ...
    getTurbulance( mean_rets, current_rets, cov)
warning('off','MATLAB:nearlySingularMatrix');
% Obtain covariance matrix from vols and correlation
n        = length(mean_rets);
cov_diag = eye(n) .* cov;

% Turbulance Factor from Kinlaw & Turkington (2012)
dt = (current_rets-mean_rets)/(cov+eye(n)*0.0001)*(current_rets-mean_rets)'/n;

% Magnitude Surprise
mag_sup = (current_rets-mean_rets)/(cov_diag)*(current_rets-mean_rets)'/n;

% Correlation Surprise
corr_sup = dt/mag_sup;

end


