
clc;
clear all;

EMat = csvread('Data/EMat.csv');
TA = csvread('Data/TA.csv');
FL = csvread("Data/FL.csv");

Nf = 5;
Nch = size(EMat,1)/Nf;


T0 = 10000; 
FLsize = 200;

for c=1:1:(size(EMat,1)/Nf)
    img = EMat(c*Nf,1);
    ppl = EMat(c*Nf,2);
    
    temp = (TA(:,1)==img) .* (TA(:,2)==ppl);
    currentTA = TA(temp==1,:);
    Length=size(currentTA,1);
    if(Length==0)
        fprintf("Not found.\n");
        return;
    end

    
    temp = (EMat(:,1)==img).*(EMat(:,2)==ppl);
    totalM = EMat(temp==1,:);
    M = (totalM(:,4:3+totalM(1,3)))';
    
    
    for trunk = 1:Length

        Map = 1*ones(1600+200*2,2000+200*2);
        
        pos = currentTA(trunk,4);
        ptrunk = trunk-1;
        while pos ==0
            pos = currentTA(ptrunk,4);
            ptrunk = ptrunk-1;
        end
        Trem = T0;
        
        for line = pos:-1:1
            if Trem == 0
                break;
            end
            
            dur = M(line,5);
            x0 = floor(M(line,2)*10.5);
            y0 = floor(M(line,1)*16.8);
            

            if y0<40 || y0>1640
                continue;
            elseif x0<25 || x0>1025
                continue;
            end
            
            dur = min(Trem, dur);
            Trem = Trem-dur;
            
            addon = filter*dur;

            Map(x0+200-FLsize:x0+200+FLsize,y0+200-FLsize:y0+200+FLsize) = ...
                Map(x0+200-FLsize:x0+200+FLsize,y0+200-FLsize:y0+200+FLsize)+addon;
            
        end
        

        shru = zeros(50,60);
        for x=1:50
            for y=1:60
                shru(x,y) = max(max( Map((x-1)*40+1:x*40,(y-1)*40+1:y*40) ));
            end
        end
        
        shru = floor(255*shru/max(max(shru)));
        

    end
    
end
