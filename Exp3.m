% Exp3 algorithm
% Input: Number of experiments, number of rounds, epsilon, mean of arms, 
% indicator that determines whether to use dynamic epsilon
% Output: All regrets for each experiment and each round

function all_regrets = Exp3(K,T,eta,miu,indicator)
    all_regrets = zeros(K,T);
    arm_num = length(miu);  % Total number of arms
    arm_index = 1:arm_num;  % Used when dealing with more than 2 arms
    % Calculate the optimal arm and its index in order to calculate the regret
    [opt_r,opt_r_index] = max(miu);
    
    % Exp3 algorithm
    for k = 1:K % repeat experiment take average

        est_rewards = zeros(1,arm_num); % Total estimated reward by the end of current round 

        round_reward = zeros(1,T);  % Save round reward in each experiment


        % start experiment

        for t = 1:T
            if indicator 
                eta = sqrt(2*log(arm_num)/(arm_num*t));
            end
            % Calculating weights using exponential, minus a constant to avoid overflow
            weights = exp(eta*(est_rewards-max(est_rewards)));

            % using softmax to calculate the probability distribution
            sum_weight = sum(weights);
            prob = weights./sum_weight; % The sum of probs must equal to 1

            % determine which arm we will choose, only for 2 arms
            if arm_num == 2
                % determine which arm we will choose, only for 2 arms
                if rand < prob(1)
                    choice = 1;
                else
                    choice = 2;
                end
            else
                % Choose the arm using probability distributions, for
                % multiple arms
                choice = randsrc(1,1,[arm_index;prob]);
            end

            % Observe the round reward for the chosen arm
            exp_ob = (rand < miu(choice));

            % Update the estimated reward for all arms
            for i=1:arm_num
                if i == choice
                    est_rewards(i) = est_rewards(i) + 1 - (1-exp_ob)/prob(i);
                else
                    est_rewards(i) = est_rewards(i) + 1;
                end
            end

            % Using mean as the round reward
            round_reward(t) = miu(choice);

        end

        % Calculate the regrets for each experiment
        all_regrets(k,:) = opt_r*(1:T) - cumsum(round_reward);

    end
end