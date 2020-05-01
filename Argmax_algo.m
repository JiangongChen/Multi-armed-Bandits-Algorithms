% Argmax algorithm
% Input: Number of experiments, number of rounds, mean of arms
% Output: All regrets for each experiment and each round

function all_regrets = Argmax_algo(K,T,miu)
    all_regrets = zeros(K,T);
    arm_num = length(miu);  % Total number of arms
    % Calculate the optimal arm and its index in order to calculate the regret
    [opt_r,opt_r_index] = max(miu);

    % Hedge algorithm with full information
    for k = 1:K % repeat experiment take average

        cumu_reward = zeros(1,arm_num); % Initialize the cumulative rewards
        exp_ob = zeros(1,arm_num); % Store the observation result for each arm in each round
%         arm_select = zeros(1,arm_num);
        
        round_reward = zeros(1,T);  % save round reward in each experiment

        % start experiment

        for t = 1:T
            if t <= arm_num
                choice = t;
            else
                % Using argmax to determine the arm choice
                choice = argmax((cumu_reward-max(cumu_reward)));
%                 choice = argmax((cumu_reward./arm_select-max(cumu_reward./arm_select)));
            end

            % Observe the results for all arms, therefore we have full
            % access to information
            for i = 1:arm_num
                exp_ob(i) = (rand < miu(i));
                cumu_reward(i) = cumu_reward(i) + exp_ob(i); % Update the losses
            end
            % Observe partial results for chosen arm
%             exp_ob = (rand < miu(choice));
%             cumu_reward(choice) = cumu_reward(choice) + exp_ob;
%             arm_select(choice) = arm_select(choice) + 1;

            % Using mean as the round reward
            round_reward(t) = miu(choice);

        end

        % Calculate the regrets for each experiment
        all_regrets(k,:) = opt_r*(1:T) - cumsum(round_reward);
    end
end