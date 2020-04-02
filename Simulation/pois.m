
function p=pois(S)

if(S<=100)
    temp=-S;
    L=exp(temp);
    k=0;
    p=1;
    while(p > L)
        k=k+1;
        p=p*rand();
    end
    p=k-1;
else
    p=floor(S+S^.5*randn());
end
end