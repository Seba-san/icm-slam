load('datos_palomar1.mat')
obs=datos.observaciones;
T=size(obs,2);

%recorto rango máximo
max_dist=10;
obs_inicial=min(obs,max_dist);
obs(obs>max_dist)=NaN;

%detecto momentos en los que entra ruido
a=[];
for t=1:T
    laser=obs(:,t);
    a=[a,size(laser(isnan(laser)==0),1)];
end
cant_max=15;
a=[a,cant_max];
plot(a,'r')

xxxxxxxxxxx


t=1:T+1;tt=t;
tt(a>cant_max)=[];
a(a>cant_max)=[];
a=fix(interp1(tt,a,t,'linear'));
a(end)=[];
hold on
plot(a)

%grafico puntos filtrados
ang=0:1:180;ang=ang'*pi/180;
for t=1:T
    laser=obs(:,t);
%     plot(laser)
%     axis([0 180 0 max_dist])

    [~,ind]=sort(laser);
    ind=ind(a(t)+1:end);
    laser(ind)=NaN;

    ptos=[cos(ang).*laser,sin(ang).*laser];
    plot(ptos(:,1),ptos(:,2),'r*')
    axis([-max_dist max_dist 0 max_dist])
    pause(0.01)
    clf
    obs(:,t)=laser;
end

obs(isnan(obs))=max_dist;
% for t=1:T
%     plot(obs_inicial(:,t))
%     hold on
%     plot(obs(:,t),'r')
%     axis([0 180 0 max_dist])
%     pause(0.1)
%     clf
% 
% end

datos.observaciones=obs;
save('datos_palomar1_filt.mat','datos')
