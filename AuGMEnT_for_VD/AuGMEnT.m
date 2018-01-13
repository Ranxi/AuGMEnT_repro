%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%    Implemention of the model AuGMEnT     %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function AuGMEnT()
Tmax = 9;
inputsize =  4;  
MemNum = 4;      
RegNum = 3;      
QvNum = 3;       
TrialNum = 4;    
trialtype = 0;   
PhaseNum = 5;    
phases_dur = [1,10,1,2,8];   
% phaseS_dur = 1;
% phaseF_dur = 10;
% phaseC_dur = 1;
% phaseD_dur = 2;
% phaseG_dur = 8;
r_i = 0.2;       
r_f = 1.5;       


%%% model parameters
bta = 0.15;              
lmd = 0.2;               
gama = 0.9;              
alpha = 1-lmd*gama;      
epsilon = 0.025;         

%%% ============= Input Layer ============= %%%

% Instant input sequence in 5 phases for a trial
seq_x_ins = zeros(PhaseNum,inputsize);
% transient on-units sequence in 5 phases for a trial
seq_x_on = zeros(PhaseNum,inputsize);
% transient off-units sequence in 5 phases for a trial
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
Qa = zeros(1,2);     
Qrecord = zeros(PhaseNum,QvNum);      
SuccQrcd = zeros(PhaseNum,QvNum);    

%%% ============= Weights matrix ============= %%%
%%% Initialize weights by scattering them over [-0.25,0.25] evenly.
v_reg = zeros(inputsize,RegNum);
v_mem = zeros(inputsize*2,MemNum);     
w = zeros(RegNum+MemNum,QvNum);

%%% ============= RPE ============= %%%
rpe = 0;

tag_reg = zeros(inputsize,RegNum);
tag_mem = zeros(inputsize*2,MemNum);
tag_A2Q = zeros(RegNum+MemNum, QvNum);
sTrace = zeros(inputsize*2, MemNum);

%%% ==== vairables for result analysis ====
AttainGoalWindow = 100;              
AttainGoalpercent = 0.99;            
results = zeros(AttainGoalWindow,1); 

% run iterations
initializeWeights();
goodStart = false;       
pnoOut = 0;              
for trial =1:80000
    fprintf("iteration #: %d\t",trial);
 
    [trialtype,seq_x_ins,seq_x_on,seq_x_off,reward] = getdata();     
    training = true;
    abortTrial = false;      
    fixSucc = 0;             
    trialSucc = false;       
    resetTrial();            
    pno = 2;                 
    cur_reward = 0;
    while training && pno <= PhaseNum
        for ts=1:phases_dur(pno)
            calculateNet(pno,ts);    
         
            chooseHighest = (rand() > epsilon);
            if(chooseHighest)
                [~,Qa(2)]=max(Qv);
            else         
                BoltzmanPb = exp(Qv);
                BoltzmanPb = BoltzmanPb/sum(BoltzmanPb);
                BoltzmanPbCum = cumsum(BoltzmanPb);
                sa = rand();
                temp = find(BoltzmanPbCum>=sa,1);
                if(isnan(temp))
                    Qa(2) = 3;   
                else
                    Qa(2) = temp;
                end
            end
         
            rpe = cur_reward + gama * Qv(Qa(2)) - Qva_pre;
            z = zeros(1,QvNum);
            z(Qa(2)) = 1;
            delta_w = bta * rpe * tag_A2Q;
            delta_v_reg = bta * rpe * tag_reg;
            delta_v_mem = bta * rpe * tag_mem;
         
            w = w+ delta_w;
            v_reg = v_reg + delta_v_reg;
            v_mem = v_mem + delta_v_mem;
            if(abortTrial)                   
                training = false;               %% terminate this trial due to fixation broken
                fprintf("Terminated!\tAT PHASE %d\t",pnoOut);
                break;
            end
            if(pno==5 && Qa(1)~=2)  
                if(trialSucc)
                    fprintf("\t\t\tSuccessed!\t TrialType:%d\t",trialtype);
                    break;
                else
                    fprintf("\tWrong Action!\t");
                    break;
                end
            end
         
            sTrace = sTrace + repmat([cur_x_on cur_x_off]',[1 MemNum]);
%             delta_tag_A2Q = -alpha * tag_A2Q + 
%             delta_tag_reg = -alpha * tag_reg + 
%             delta_tag_mem = -alpha * tag_mem + 
            tag_reg = lmd*gama*tag_reg + (cur_x_ins' * y_r.*(1-y_r)).* repmat(w(1:RegNum,Qa(2))',[inputsize 1]);
            tag_mem = lmd*gama*tag_mem + sTrace .* repmat((y_m.*(1-y_m).* w(RegNum+1:RegNum+MemNum,Qa(2))'),[inputsize*2 1]);
            tag_A2Q = lmd*gama*tag_A2Q + [y_r y_m]' * z;
            Qa(1) = Qa(2);
            Qva_pre = Qv(Qa(2));
         
            if(2 < pno && pno < 5 && Qa(2)~=2)   
                abortTrial = true;
                pnoOut = pno;
            else
                if(2 == pno)                     
                    fixSucc = (fixSucc+1)*(Qa(2)==2);    
                    goodStart = goodStart || (Qa(2)==2);
                    if(ts==phases_dur(2) && Qa(2)~=2)
                        abortTrial = true;
                        pnoOut = pno;
                    end
                end
            end
            if(pno < 5)
                cur_reward = r_i * reward(pno,Qa(2)) * (fixSucc>1);    
            else
                cur_reward = r_f * reward(pno,Qa(2));
                trialSucc = (cur_reward>0);
            end
            if(pno==2 && fixSucc>=2)
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
    fprintf("\n");
end
fprintf("TrialType:%d\n",trialtype);
figure
p = plot(SuccQrcd);
p(1).Marker = '*';
p(2).Marker = '+';
p(3).Marker = 'o';

    function calculateNet(pno,t)
        %parameter
     
     
        cur_x_ins = seq_x_ins(pno,:);
        if(t==1)
            cur_x_on = seq_x_on(pno,:);
            cur_x_off = seq_x_off(pno,:);
        else
            if(t <= phases_dur(pno))
                cur_x_on = zeros(1,inputsize);
                cur_x_off = zeros(1,inputsize);
            end
        end
        y_r = cur_x_ins * v_reg;
        y_r = 1./(1+exp(-y_r));
        inp_m_pre = inp_m_pre + [cur_x_on cur_x_off] * v_mem;
        y_m = 1./(1+exp(-inp_m_pre));
        Qv = [y_r y_m] * w;
    end
    
    function initializeWeights()
        v_reg = (rand(inputsize,RegNum)-0.5)/2;
        v_mem = (rand(inputsize*2,MemNum)-0.5)/2;     
        w = (rand(RegNum+MemNum,QvNum)-0.5)/2;
        fprintf("Regenerate Weights.\t");
    end
    
    function resetTrial()
        Qv = zeros(1,QvNum);
        Qva_pre = 0;
        Qa = zeros(1,2);
        inp_m_pre = zeros(1,MemNum);
        tag_reg = zeros(inputsize,RegNum);
        tag_mem = zeros(inputsize*2,MemNum);
        tag_A2Q = zeros(RegNum+MemNum, QvNum);
        sTrace = zeros(inputsize*2, MemNum);
        rpe = 0;
    end

    function sa = rand_time()
     
        time = second(now);
        sa = time - fix(time);
    end
end
