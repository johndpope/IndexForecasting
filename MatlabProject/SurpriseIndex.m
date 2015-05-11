clc; clear all;
%% Read All Data (run DataScript.m first)
filename = 'project_vars.mat';
load(filename);

% ONLY USE SUB INDEX
surprises = getCorrSurprise(daily_ret(:,12:end), 3)


%% Plot Surprise Index
mags = surprises(:,2);
corrs = surprises(:,3);
x_vals = tiedrank(mags) / length(mags);
y_vals = tiedrank(corrs) / length(corrs);
scatter(x_vals,y_vals);
h = lsline;
set(h,'color','r','LineWidth',3)
title('S&P500 Sectors Surprise Factors')
xlabel('Magnitude')
ylabel('Correlation')

%% Surf best returns
num_values = length(surprises);
x = x_vals(1:end-1);
y = y_vals(1:end-1);
z = daily_ret(end-num_values+2:end,1);
tri = delaunay(x,y);

trisurf(tri,x,y,z)

%%

[xx,yy]=meshgrid(0:0.025:1,0:0.025:1);
zz = griddata(x,y,z,xx,yy);
% zz = smooth3(zz)
surf(xx,yy,zz);

%%  
I = find(abs(z)>=0.01);

pointsize = 10;
scatter(x(I), y(I), pointsize, z(I));



