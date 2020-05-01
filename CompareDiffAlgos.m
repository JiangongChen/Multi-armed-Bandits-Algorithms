% Comparison of different algorithms for Multi-armed bandit problem 

%  Description:
%  Choose one of several arms to play, based on the historical (regard t,
%  total T) observation to estimate reward of each arm.
%  Six candidate algorithms tested:
%  1. Hedge algorithm with full information
%  2. Exp3 algorithm
%  3. basic UCB
%  4. Exp3 algorithm with 2 feedback
%  5. Argmax algorithm with full information
%  6. Hedge algorithm with 2 feedback

clear;
tic;

% define global variables
K = 5e3;    % Total number of experiments
T = 1e4;    % Total rounds of play in each experiment

% Mean of two feedback arms
theta = [0.75,0.9];
gama = [0.2,0.6];
miu = theta.*gama;   % Mean of Bernoulli arms, you can choose any number of arms.
% All of the value must be between (0,1). For example, [0.1,0.5,0.9]

% Define the hyperparameters for each algorithm
alp = 2;    % alpha used in UCB algorithm
% Indicators which determine whether to use dynamic eta in Hedge and
% Exp3 algorithms, 1 means on, 0 means off
eta_hedge_dynamic = 1;
eta_hedge_2fed_dynamic = 1;
eta_exp3_dynamic = 1;
eta_exp3_2fed_dynamic = 1;

% Fixed epsilon used in Hedge algorithm
% Notice that if the value is too high, overflow is inevitable and the
% result will be wrong
eta_hedge = 0.1;    
eta_hedge_2fed = 0.1;
% Fixed epsilon used in Exp3 algorithm
eta_exp3 = 0.02;    
eta_exp3_2fed = 0.02;

% Using flag to choose which algorithm you will use
% 1 refers to Hedge, 2 refers to Exp3, 3 refers to UCB, 4 refers to Exp3
% with 2 feedback, 5 refres to Argmax, 6 refers to Hedge with 2 feedback
% Use the combination of multiple flags to draw several lines together
% Follow the format like [1,2,3]
flags = [1,2,3];

figure()
hold on 
grid on
% Create a string array to store labels
label = [];

% Get simulation results
for i=1:length(flags)
    flag = flags(i);
    if flag == 1
        all_regrets = Hedge(K,T,eta_hedge,miu,eta_hedge_dynamic);
        label = [label "Hedge"];
    elseif flag == 2
        all_regrets = Exp3(K,T,eta_exp3,miu,eta_exp3_dynamic);
        label = [label "Exp3"];
    elseif flag == 3
        all_regrets = UCB(K,T,alp,miu);
        label = [label "UCB"];
    elseif flag == 4
        all_regrets = Exp3_2fed(K,T,eta_exp3_2fed,theta,gama,eta_exp3_2fed_dynamic);
        label = [label "Exp3 2feedback"];
    elseif flag == 5
        all_regrets = Argmax_algo(K,T,miu);
        label = [label "Argmax"];
    elseif flag == 6
        all_regrets = Hedge_2fed(K,T,eta_hedge_2fed,theta,gama,eta_hedge_2fed_dynamic);
        label = [label "Hedge 2feedback"];
    else
        error('Please change the value of flag to an integer between 1 to 5!');
    end
    % Calculate the mean regrets and plot the results
    ave_regrets = zeros(1,T);
    for j=1:T
        ave_regrets(j) = mean(all_regrets(:,j));
    end    
    plot(1:T, ave_regrets, 'LineWidth', 2);    
end



xlabel('Number of rounds')
ylabel('Expected regret')
title('Comparison of different MAB algorithms')
legend(label)

t = toc;
used_time = sprintf('Time used: %4.3f s',t);
disp(used_time)