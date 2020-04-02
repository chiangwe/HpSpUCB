function [times x y  offspring clusterId]=Hawkes_Simulation(mu,k0,w,T,sig)
%this program simulates event times, called "times", according
%to a self-exciting point process with exponential triggering kernel and
%parameters mu (const. background rate)
%k0 (branching ratio) and w (exp parameter) on the time interval [0,T]
p=pois(mu*T);

times=zeros(p,1);
x=zeros(p,1);
y=zeros(p,1);
offspring=zeros(p,1);
clusterId=zeros(p,1);
%first simulate "background" events
%this is done by picking p points where p is Poisson with parameter mu*T
%and then distributing the points uniformly in the interval [0,T]


times(1:p,1)=rand(p,1)*T;
%disp(times(1:p,1)/18000)
x(1:p,1)=rand(p,1);
y(1:p,1)=rand(p,1);
clusterId(1:p,1) = (1:p);

counts=1;
countf=p;
%display(['Num of immmigrant: ' num2str(p) ]);

%Next loop through every event and simulate the "offspring"
%even the offspring events can generate their own offspring

while((countf-counts)>-1)
    p=pois(k0); %each event generates p offspring according to a Poisson r.v. with parameter k0
    %display(p)
    
    for j=1:p
        temp=times(counts)-log(rand())/w; % this generates an exponential r.v. on [t_counts,infty]
        %-log(rand())/w
        temp2=x(counts)+sig*randn(); % inter-point distances are gaussian
        temp3=y(counts)+sig*randn();
        
        
        if(temp<T)                        % we only keep this time if it is in [t_counts,T]
            countf=countf+1;
            times(countf)=temp;
            x(countf)=temp2;
            y(countf)=temp3;
            offspring(countf)=1;
            clusterId(countf) = clusterId(counts);
        else
        end
    end
    %display(mean(meanP) )
    counts=counts+1;
    %display((countf-counts))
end
%data=[times(1:countf) x(1:countf) y(1:countf) offspring(1:countf)  clusterId(countf)];
data=[times x y offspring  clusterId];
data=sortrows(data,1);

id = find(  (data(:,2)>0) & (data(:,2)<1) & (data(:,3)>0) & (data(:,3)<1) );
data = data(id, :);

times=data(:,1);
x=data(:,2);
y=data(:,3);
offspring=data(:,4);
clusterId=data(:,5);

end


