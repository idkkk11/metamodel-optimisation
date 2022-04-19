data = csvread('optidata.csv'); % Reads from CSV file
input = data(:,1:4); % Design Variables
stress = data(:,5); % Stress Data
deflection = data(:,6); % Deformation Data
mass = data(:,7); % Mass Data

rng(1);                                 % MATLAB random seed 1
newInd = randperm(length(input));       % Creates an index parameter to shuffle data

input_new = input(newInd,:);      % Shuffled input data
stress_new = stress(newInd);        % Shuffled maximum principal stress data
deflection_new = deflection(newInd,:);  % Shuffled maximum displacement data
mass_new = mass(newInd);    % Shuffled mass data

splitPt1 = floor(0.75*length(input)); % Calculates 75% of the number of elements rounded down to the nearest integer

input_newTrain1 = input_new(1:splitPt1,:)'; % Creates the training array of "input" data
stress_newTrain1 = stress_new(1:splitPt1)'; % Creates the training array of "stress" data
deflection_newTrain1 = deflection_new(1:splitPt1)'; % Creates the training array of "deformation" data
mass_newTrain1 = mass_new(1:splitPt1)'; % Creates the training array of "mass" data

input_newTest1 = input_new(splitPt1+1:end,:)'; % Creates the test array of "input" data
stress_newTest1 = stress_new(splitPt1+1:end)'; % Creates the test array of "stress" data
deflection_newTest1 = deflection_new(splitPt1+1:end,:)'; % Creates the test array of "deformation" data
mass_newTest1 = mass_new(splitPt1+1:end)'; % Creates the test array of "mass" data

input_newer = input_new(1:length(input_new),:)'; % Creates the array of "input" data using test and train elements
deflection_newer = deflection_new(1:length(deflection_new),:)'; % Creates the array of "deformation" data using test and train elements
mass_newer = mass_new(1:length(mass_new),:)'; % Creates the array of "mass" data using test and train elements

% CASE 2 SPLIT: CEIL(0.75*N) SAME AS SECTION ABOVE BUT ROUNDS UP INSTEAD OF
% DOWN

splitPt2 = ceil(0.75*length(input));  % Case 2 split point

input_newTrain2 = input_new(1:splitPt2,:)';
stress_newTrain2 = stress_new(1:splitPt2)';
deflection_newTrain2 = deflection_new(1:splitPt2,:)';
mass_newTrain2 = mass_new(1:splitPt2)';

input_newTest2 = input_new(splitPt2+1:end,:)';
stress_newTest2 = stress_new(splitPt2+1:end)';
deflection_newTest2 = deflection_new(splitPt2+1:end,:)';
mass_newTest2 = mass_new(splitPt2+1:end)';


% LINEAR REGRESSION MODELS, CASE 1
beta1 = mvregress(input_newTrain1',stress_newTrain1');
beta2 = mvregress(input_newTrain1',mass_newTrain1');
beta3 = mvregress(input_newTrain1',deflection_newTrain1');

% LINEAR REGRESSION MODELS, CASE 2
beta4 = mvregress(input_newTrain2',stress_newTrain2');
beta5 = mvregress(input_newTrain2',mass_newTrain2');
beta6 = mvregress(input_newTrain2',deflection_newTrain2');

% SECOND ORDER POLYNOMIAL REGRESSION MODELS, CASE 1
p_stress1 = polyfitn(input_newTrain1',stress_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');
p_mass1 = polyfitn(input_newTrain1',mass_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');
p_deflection1 = polyfitn(input_newTrain1',deflection_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');

% SECOND ORDER POLYNOMIAL REGRESSION MODELS, CASE 2
p_stress2 = polyfitn(input_newTrain2',stress_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');
p_mass2 = polyfitn(input_newTrain2',mass_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');
p_deflection2 = polyfitn(input_newTrain2',deflection_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');

% THIRD ORDER POLYNOMIAL REGRESSION MODELS, CASE 1
p3_stress1 = polyfitn(input_newTrain1',stress_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');
p3_mass1 = polyfitn(input_newTrain1',mass_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');
p3_deflection1 = polyfitn(input_newTrain1',deflection_newTrain1','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');

% THIRD ORDER POLYNOMIAL REGRESSION MODELS, CASE 2
p3_stress2 = polyfitn(input_newTrain2',stress_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');
p3_mass2 = polyfitn(input_newTrain2',mass_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');
p3_deflection2 = polyfitn(input_newTrain2',deflection_newTrain2','constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');

%THIRD AND SECOND ORDER POLYNOMIAL REGRESSION MODELS USING BOTH FIRST AND
%SECOND CASES
p3_deflection = polyfitn(input_newer', deflection_newer', 'constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2,x1^3,x2^3,x3^3,x4^3');
p3_mass = polyfitn(input_newer', mass_newer', 'constant,x1,x2,x3,x4,x1^2,x2^2,x3^2,x4^2');




% R2 VALUES, CASE 1 Test
Rsq_1 = 1 - norm(input_newTest1'*beta1 - stress_newTest1')^2/norm(stress_newTest1-mean(stress_newTest1))^2;
Rsq_2 = 1 - norm(input_newTest1'*beta2 - mass_newTest1')^2/norm(mass_newTest1-mean(mass_newTest1))^2;
Rsq_3 = 1 - norm(input_newTest1'*beta2 - deflection_newTest1')^2/norm(deflection_newTest1-mean(deflection_newTest1))^2;
% R2 VALUES, CASE 2 Test
Rsq_4 = 1 - norm(input_newTest2'*beta4 - stress_newTest2')^2/norm(stress_newTest2-mean(stress_newTest2))^2;
Rsq_5 = 1 - norm(input_newTest2'*beta6 - mass_newTest2')^2/norm(mass_newTest2-mean(mass_newTest2))^2;
Rsq_6 = 1 - norm(input_newTest2'*beta6 - deflection_newTest2')^2/norm(deflection_newTest2-mean(deflection_newTest2))^2;

% R2 VALUES, CASE 1 Train
R2_1 = 1 - norm(input_newTrain1'*beta1 - stress_newTrain1')^2/norm(stress_newTrain1-mean(stress_newTrain1))^2;
R2_2 = 1 - norm(input_newTrain1'*beta2 - mass_newTrain1')^2/norm(mass_newTrain1-mean(mass_newTrain1))^2;
R2_3 = 1 - norm(input_newTrain1'*beta2 - deflection_newTrain1')^2/norm(deflection_newTrain1-mean(deflection_newTrain1))^2;
% R2 VALUES, CASE 2 Train
R2_4 = 1 - norm(input_newTrain2'*beta4 - stress_newTrain2')^2/norm(stress_newTrain2-mean(stress_newTrain2))^2;
R2_5 = 1 - norm(input_newTrain2'*beta6 - mass_newTrain2')^2/norm(mass_newTrain2-mean(mass_newTrain2))^2;
R2_6 = 1 - norm(input_newTrain2'*beta6 - deflection_newTrain2')^2/norm(deflection_newTrain2-mean(deflection_newTrain2))^2;

%RMSE VALUES
Rmse_1 = sqrt(norm(input_newTrain1'*beta1 - stress_newTrain1')^2/length(stress_newTrain1));
Rmse_2 = sqrt(norm(input_newTrain1'*beta2 - mass_newTrain1')^2/length(mass_newTrain1));
Rmse_3 = sqrt(norm(input_newTrain1'*beta3 - deflection_newTrain1')^2/length(deflection_newTrain1));
Rmse_4 = sqrt(norm(input_newTrain2'*beta4 - stress_newTrain2')^2/length(stress_newTrain2));
Rmse_5 = sqrt(norm(input_newTrain2'*beta5 - mass_newTrain2')^2/length(mass_newTrain2));
Rmse_6 = sqrt(norm(input_newTrain2'*beta6 - deflection_newTrain2')^2/length(deflection_newTrain2));

%RMSE VALUES
Rmse_1 = sqrt(norm(input_newTest1'*beta1 - stress_newTest1')^2/length(stress_newTest1));
Rmse_2 = sqrt(norm(input_newTest1'*beta2 - mass_newTest1')^2/length(mass_newTest1));
Rmse_3 = sqrt(norm(input_newTest1'*beta3 - deflection_newTest1')^2/length(deflection_newTest1));
Rmse_4 = sqrt(norm(input_newTest2'*beta4 - stress_newTest2')^2/length(stress_newTest2));
Rmse_5 = sqrt(norm(input_newTest2'*beta5 - mass_newTest2')^2/length(mass_newTest2));
Rmse_6 = sqrt(norm(input_newTest2'*beta6 - deflection_newTest2')^2/length(deflection_newTest2));

%Optimisation using the Genetic Algorithm Method

clear all; % Clears the matlab workspace

% Sets optimisation options where the solver used is 'ga' for genetic
% algorithm. ConstraintTolerance sets the upper bound to the constraint
% function to which the algorithm stops. PlotFcn plots the iterative graph
% as the algorithm is moving.
options = optimoptions ('ga', 'ConstraintTolerance', 1e-6, 'PlotFcn',@gaplotbestf); 

% This variable sets the objective function using the coefficients
% previously derived from the model, taken from the mass model.
fun = @(x) 5.04825880596942 - 24.8924159604721*x(1) - 30.7226312130383*x(2) + 4.45095862593356*x(3) + 566.100222767926*x(4) + 38.7851697547828*x(1)^2 + 51.2648237602782*x(2)^2+ 2.57642095453503*x(3)^2 + 11252.1140845118*x(4)^2;

% nvar states the number of variables to optimise with lb and ub being the
% lower and upper bound of each variable respectively. A,b,Aeq, and beq is
% empty as there aren't any linear constraints. xga is set as the variable
% to receive the output from the genetic algorithm. minimumga prints this
% value onto the command window.
nvar = 4;
lb = [0.2,0.2,0.2,0.001];
ub = [0.6,0.6,0.6,0.02];
A = [];
b = [];
Aeq = [];
beq = [];
xga = ga(fun,nvar,A,b,Aeq,beq,lb,ub,@nonlcon,options);
minimumga = xga

% this section is where the non-linear constraints are added where c(1) and
% c(2) are the constraints on the deformation model to make sure it is
% below 10mm. c(3) is the constraint to make sure the volume is above
% 0.0115 m3.
function [c,ceq] = nonlcon(x)
    c(1) = 29.18940272 + 320.5700876*x(1) + 1104.125003*x(2) - 1234.541253*x(3) - 13338.1269*x(4) - 1200.981463*x(1)^2 - 3008.078602*x(2)^2 + 3342.317576*x(3)^2 + 1433570.522*x(4)^2 + 1216.313197*x(1)^3	+ 2672.427452*x(2)^3 - 2959.362374*x(3)^3 - 45947615.76*x(4)^3;
    c(2) = -1*(29.18940272 + 320.5700876*x(1) + 1104.125003*x(2) - 1234.541253*x(3) - 13338.1269*x(4) - 1200.981463*x(1)^2 - 3008.078602*x(2)^2 + 3342.317576*x(3)^2 + 1433570.522*x(4)^2 + 1216.313197*x(1)^3	+ 2672.427452*x(2)^3 - 2959.362374*x(3)^3 - 45947615.76*x(4)^3);
    c(3) = 1 - (x(1)*x(2)*x(3))/0.0115;
    ceq = [];
end


%Optimisation using the SQP Method

% Sets optimisation options where the solver used is 'fmincon'. 
% The iterations are set to be displayed, and the optimisation algorithm
% is set to be SQP. The step tolerance is 1e-6, sets the termination
% requirements.
options = optimoptions ('fmincon', 'Display', 'iter', 'Algorithm','sqp','StepTolerance',1e-6);

% This variable sets the objective function using the coefficients
% previously derived from the model, taken from the mass model.
fun = @(x) 5.04825880596942 - 24.8924159604721*x(1) - 30.7226312130383*x(2) + 4.45095862593356*x(3) + 566.100222767926*x(4) + 38.7851697547828*x(1)^2 + 51.2648237602782*x(2)^2+ 2.57642095453503*x(3)^2 + 11252.1140845118*x(4)^2;

% nvar states the number of variables to optimise with lb and ub being the
% lower and upper bound of each variable respectively. A,b,Aeq, and beq is
% empty as there aren't any linear constraints. x is set as the variable
% to receive the outputs from the SQP algorithm. minimum prints this
% value onto the command window.
x0 = [0.4,0.4,0.4,0.01];
lb = [0.2,0.2,0.2,0.001];
ub = [0.6,0.6,0.6,0.02];
A = [];
b = [];
Aeq = [];
beq = [];
x = fmincon(fun,x0,A,b,Aeq,beq,lb,ub,@nonlcon,options);
minimum = x

% this section is where the non-linear constraints are added where c(1) and
% c(2) are the constraints on the deformation model to make sure it is
% below 10mm. c(3) is the constraint to make sure the volume is above
% 0.0115 m3.
function [c,ceq] = nonlcon(x)
    c(1) = 29.18940272 + 320.5700876*x(1) + 1104.125003*x(2) - 1234.541253*x(3) - 13338.1269*x(4) - 1200.981463*x(1)^2 - 3008.078602*x(2)^2 + 3342.317576*x(3)^2 + 1433570.522*x(4)^2 + 1216.313197*x(1)^3	+ 2672.427452*x(2)^3 - 2959.362374*x(3)^3 - 45947615.76*x(4)^3;
    c(2) = -1*(29.18940272 + 320.5700876*x(1) + 1104.125003*x(2) - 1234.541253*x(3) - 13338.1269*x(4) - 1200.981463*x(1)^2 - 3008.078602*x(2)^2 + 3342.317576*x(3)^2 + 1433570.522*x(4)^2 + 1216.313197*x(1)^3	+ 2672.427452*x(2)^3 - 2959.362374*x(3)^3 - 45947615.76*x(4)^3);
    c(3) = 1 - ((x(1)*x(2)*x(3))/0.0115);
    ceq = [];
end