%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    Implemention of the model AuGMEnT     %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AuGMEnT()
% Tmax = 9;
inputsize =  21;     % it varys when task changes
MemNum = 4;
RegNum = 3;
QvNum = 3;
%TrialNum = 4;
%trialtype = 0;
PhaseNum = 4;
phases_dur = [10,1,2,8];      % duration of Five phases
% phaseS_dur = 10;
% phaseF1_dur = 1;
% phaseD_dur = 2;
% phaseF2_dur = 8;

r_i = 0.2;
r_f = 1.5;


%%% model parameters
bta = 0.15;                 % learning rate
lmd = 0.2;                  % Tag/Trace decay rate
gama = 0.9;                 % Discount factor
alpha = 1-lmd*gama;         % Tag persistence
epsilon = 0.025;            % Exploration rate

%%% ============= Input Layer ============= %%%
theta_c = [9.5:4:45.5];
theta_c = [theta_c;theta_c];
theta_c = theta_c(:)';
steepness =5* (-1).^(1:inputsize-1);
x_resp = zeros(1,inputsize-1);              % the first unit is binary input 'S' represents the contact with vibrating probe
F1 = 0;
F2 = 0;

% Instant input sequence in 5 phases
seq_x_ins = zeros(PhaseNum,inputsize);
% transient on-units sequence in 5 phases
seq_x_on = zeros(PhaseNum,inputsize);
% transient off-units sequence in 5 phases
seq_x_off = zeros(PhaseNum,inputsize);
% -------- Current input data ------------ %
cur_x_ins = zeros(1,inputsize);
cur_x_on = zeros(1,inputsize);
cur_x_off = zeros(1,inputsize);

%%% ============= Association Layer ============= %%%
% Regular units
y_r = zeros(1,RegNum);
% Memory units
inp_m_pre = zeros(1,MemNum);
y_m = zeros(1,MemNum);

%%% ============= Q-value Layer ============= %%%
Qv = zeros(1,QvNum);
Qva_pre = 0;
Qa = zeros(1,2);        % Qa(1): previous Q-action,     Qa(2):current Q-action
Qrecord = zeros(PhaseNum-1,QvNum);
SuccQrcd = zeros(PhaseNum-1,QvNum);

%%% ============= Weights matrix ============= %%%
%%% Initialize weights by scattering them over [-0.25,0.25] evenly.
v_reg = zeros(inputsize,RegNum);
v_mem = zeros(inputsize*2,MemNum);        % inputsize*2£¬combinate the on- and off-units
w = zeros(RegNum+MemNum,QvNum);

%%% ============= reward vector ============== %%%
%reward = zeros(PhaseNum,QvNum);
%cur_reward = 0;

%%% ============= RPE vector ============= %%%
rpe = 0;

tag_reg = zeros(inputsize,RegNum);
tag_mem = zeros(inputsize*2,MemNum);
tag_A2Q = zeros(RegNum+MemNum, QvNum);
sTrace = zeros(inputsize*2, MemNum);

%%% ==== vairables for result analysis ====
AttainGoalpercent = 0.99;
AttainGoalWindow = 100;
results = zeros(AttainGoalWindow,1);

originW_v_reg = zeros(inputsize,RegNum);
originW_v_mem = zeros(inputsize*2,MemNum);
originW_w = zeros(RegNum+MemNum,QvNum);

% run iterations
initializeWeights();
goodStart = false;
pnoOut = 0;
for trial =1:250000
    fprintf('iteration : %d\t',trial);
    % show empty screen for one time step
    [F1,F2,seq_x_ins,seq_x_on,seq_x_off,reward] = getdata();
    training = true;
    abortTrial = false;
    fixSucc = 0;
    trialSucc = false;
    resetTrial();
    pno = 1;
    cur_reward = 0;
    while training && pno <= PhaseNum
        %rand('seed',rand_time);
        compute_input(pno);
        for ts=1:phases_dur(pno)
            calculateNet(ts);
            Qv = round(Qv,5);
            % select action
            chooseHighest = (rand() > epsilon);
            if(chooseHighest)
                [~,Qa(2)]=max(Qv);
            else
                fprintf('\t\tExplore--\t');
                BoltzmanPb = exp(Qv);
                BoltzmanPb = BoltzmanPb/sum(BoltzmanPb);
                BoltzmanPbCum = cumsum(BoltzmanPb);
                sa = rand();
                temp = find(BoltzmanPbCum>=sa,1);
                if(isempty(temp))
                    Qa(2)=3;
                else
                    Qa(2) = temp;
                end
            end
            % compute RPE (use the reward for Qa(t-1))
            if(isnan(Qva_pre))
                Qva_pre = Qv(Qa(2));
            end
            rpe = cur_reward + gama * Qv(Qa(2)) - Qva_pre;
            z = zeros(1,QvNum);
            z(Qa(2)) = 1;
            delta_w = bta * rpe * tag_A2Q;
            delta_v_reg = bta * rpe * tag_reg;
            delta_v_mem = bta * rpe * tag_mem;
            % update weights
            w = w+ delta_w;
            v_reg = v_reg + delta_v_reg;
            v_mem = v_mem + delta_v_mem;
            if(abortTrial)
                training = false;               %% terminate this trial due to fixation broken
                fprintf('Terminated!\tAT PHASE %d\t',pnoOut);
                break;
            end
            if(pno==4 && (trialSucc||Qa(1)~=2))
                if(trialSucc)
                    fprintf('\t\t\tSuccessed!\tF1:%f,F2:%f\t',F1,F2);
                    break;
                else
                    fprintf('\tWrong Action!\t');
                    break;
                end
            end
            % update traces firstly, and then update tags   ||||||||||||||   + (cur_x_ins * (y_r.*(1-y_r))).* 
            sTrace = sTrace + repmat([cur_x_on cur_x_off]',[1 MemNum]);
%             delta_tag_A2Q = -alpha * tag_A2Q + [y_r y_m]' * z;
%             delta_tag_reg = -alpha * tag_reg + repmat(w(1:RegNum,Qa(2))',[inputsize 1]);
%             delta_tag_mem = -alpha * tag_mem + sTrace .* repmat((y_m.*(1-y_m).* w(RegNum+1:RegNum+MemNum,Qa(2))'),[inputsize*2 1]);
            tag_reg = lmd*gama*tag_reg + (cur_x_ins' * (y_r.*(1-y_r))).* repmat(w(1:RegNum,Qa(2))',[inputsize 1]);
            tag_mem = lmd*gama*tag_mem + sTrace .* repmat((y_m.*(1-y_m).* w(RegNum+1:RegNum+MemNum,Qa(2))'),[inputsize*2 1]);
            tag_A2Q = lmd*gama*tag_A2Q + [y_r y_m]' * z;
            Qa(1) = Qa(2);
            Qva_pre = Qv(Qa(2));
            % check whether the model has learned to fixate
            if(1 < pno && pno < 4 && Qa(2)~=2)
                abortTrial = true;
                pnoOut = pno;
            else
                if(1 == pno)
                    fixSucc = (fixSucc+1)*(Qa(2)==2);
                    goodStart = goodStart|| (Qa(2)==2);
                    if(ts==phases_dur(1) && Qa(2)~=2)
                        abortTrial = true;
                        pnoOut = pno;
                    end
                end
            end
            % get the reward
            if(pno < 4)
                cur_reward = r_i * reward(pno,Qa(2)) * (fixSucc>1);       % only when pno==1&&fixSucc>=2, we give it reward r_i
            else
                cur_reward = r_f * reward(pno,Qa(2));
                trialSucc = (cur_reward>0);
            end
            if(pno==1 && fixSucc>=2)
                break;
            end
        end
        Qrecord(pno,:) = Qv;
        pno = pno + 1;
    end
    if(trialSucc)
        results(mod(trial,100)+1) = 1;
        SuccQrcd = Qrecord;
    else
        results(mod(trial,100)+1) = 0;
    end
    if(trial>100 && sum(results)/AttainGoalWindow > AttainGoalpercent)
        bta = 0;
        epsilon = 0;
        break;
    end
    if(~goodStart)
        initializeWeights();
    end
    fprintf('\n');
end
figure
p = plot(SuccQrcd);
p(1).Marker = '*';
p(2).Marker = '+';
p(3).Marker = 'o';
% originW_v_reg
% originW_v_mem
% originW_w

    function compute_input(pno)
        % compute input at the first time step of each phase
        if(pno==1)          % Start phase
            getCurve(F1);
        else 
            if(pno==4)      % F2 phase
                getCurve(F2);
            end
        end
        cur_x_ins = seq_x_ins(pno,:) .* [1,x_resp];
        cur_x_on = seq_x_on(pno,:) .* [1,x_resp];
        cur_x_off = seq_x_off(pno,:) .* [1,x_resp];
    end

    function calculateNet(t)
        %parameter: pno, phase sequence number; t, time-step in this phase
        if(t > 1)
            cur_x_on = zeros(1,inputsize);
            cur_x_off = zeros(1,inputsize);
        end
        y_r = cur_x_ins * v_reg;
        y_r = 1./(1+exp(-y_r));
        inp_m_pre = inp_m_pre + [cur_x_on cur_x_off] * v_mem;
        y_m = 1./(1+exp(-inp_m_pre));
        Qv = [y_r y_m] * w;
    end
    
    function initializeWeights()
        v_reg = (rand(inputsize,RegNum)-0.5)/2;
        v_mem = (rand(inputsize*2,MemNum)-0.5)/2;        % inputsize*2£¬combinate the on- and off-units
        w = (rand(RegNum+MemNum,QvNum)-0.5)/2;
        originW_v_reg = v_reg;
        originW_v_mem = v_mem;
        originW_w = w;
        fprintf('Regenerate Weights.\t');
    end
    
    function resetTrial()
        Qv = zeros(1,QvNum);
        Qva_pre = NaN;
        Qa = zeros(1,2);
        inp_m_pre = zeros(1,MemNum);
        tag_reg = zeros(inputsize,RegNum);
        tag_mem = zeros(inputsize*2,MemNum);
        tag_A2Q = zeros(RegNum+MemNum, QvNum);
        sTrace = zeros(inputsize*2, MemNum);
        rpe = 0;
    end

    function getCurve(f)
        freq_array = f*ones(1,inputsize-1);
        offset = theta_c - freq_array;
        x_resp =  single(offset .* steepness);
        x_resp = 1 ./ (1+exp(x_resp));
    end
    
end
