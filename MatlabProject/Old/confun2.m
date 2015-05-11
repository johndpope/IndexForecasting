function [c, ceq] = confun2(w, covar, target_risk, daily_rets, targetDD)
% 200% >= sp500 + bonds + mom >= 100%
PortfolioRisk    = @(w,cov) (w*cov*w');
maxDD            = max(DrawDown(w, daily_rets));

% Nonlinear inequality constraints
c   =  [
       (w(3) + w(4) - 1);
       (sqrt(PortfolioRisk(w,covar)) - target_risk);
       (maxDD - targetDD);
       1 - (w(1) + w(2) + w(5));
       (w(1) + w(2) + w(5)) - 2];

% Nonlinear equality constraints
ceq = [];

end