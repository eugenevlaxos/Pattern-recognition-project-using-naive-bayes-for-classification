%%Part A: Data Generation

close all;clf;clc;clear;

figure(1);
title("Clusters");
grid on;
hold on;

N1 = 400;
N2 = 100;

x1 = 2 + 6.*rand(1,N1);
x2 = 1 + 1.*rand(1,N1);
w1 = [x1;x2];

x1 = 6 + 2.*rand(1,N2);
x2 = 2.5 + 3.*rand(1,N2);
w2 = [x1;x2];


scatter(w1(1,:), w1(2,:), "*b");
scatter(w2(1,:), w2(2,:), "*r");
legend("w1","w2",'Location','northwest');

%%Part B: Bayesian Classification in 2-D Space 
figure(2);
title("maximum likelihood estimation");
hold on;

m1=mean(w1');
m2=mean(w2');

S1=(w1-ones(1,N1).*m1')*(w1-ones(1,N1).*m1')'/N1;
S2=(w2-ones(1,N2).*m2')*(w2-ones(1,N2).*m2')'/N2;

f1=0:0.1:10;
f2=0:0.1:10;

[F1,F2]=meshgrid(f1,f2);
F=[F1(:) F2(:)];

mle=mvnpdf(F,m1,S1);
mle=reshape(mle,length(f2),length(f1));
surf(f1,f2,mle)

mle=mvnpdf(F,m2,S2);
mle=reshape(mle,length(f2),length(f1));
surf(f1,f2,mle)

figure(3);
title("Least distance of classifiers");
hold on;
A=[w1 w2];

k=0;
p=0;

for i=1:(N1+N2)
    res1=norm(A(:,i)-m1');
    res2=norm(A(:,i)-m2');
    
    if(res1<res2)
        k=k+1;
        w1_new(:,k)=A(:,i);
    else
        p=p+1;
        w2_new(:,p)=A(:,i);  
    end
end  
  
scatter(w1_new(1,:), w1_new(2,:), "*b");
scatter(w2_new(1,:), w2_new(2,:), "*r");

limit=2; %% discriminant function:x2=2
m=0;
n=0;

w1_error=[0;0];
w2_error=[0;0];
for i=1:k
    if(w1_new(2,i)>limit)
        m=m+1;
        w1_error(:,m)=w1_new(:,i);
    end
end

for i=1:p
    if(w2_new(2,i)<limit)
        n=n+1;
        w2_error(:,n)=w2_new(:,i);
    end
end

error_percent=((m+n)/(N1+N2))*100;
fprintf("error percentage=: %.3d%%\n",error_percent);

scatter(w1_error(1,:), w1_error(2,:), "*m");
scatter(w2_error(1,:), w2_error(2,:), "*g");
legend("w1","w2","w1 error","w2 error",'Location','northwest');

figure(4);
title("least Mahalanobis distance of classifiers");
hold on;

P_w1=N1/N1+N2;
P_w2=N2/N1+N2;

S_m=P_w1*S1 + P_w2*S2;
S_m_inv=inv(S_m);

k=0;
p=0;

for i=1:(N1+N2)
    res1=norm((A(:,i)-m1')'*S_m_inv*(A(:,i)-m1'));
    res2=norm((A(:,i)-m2')'*S_m_inv*(A(:,i)-m2'));
    
     if(res1<res2)
        k=k+1;
        w1_m(:,k)=A(:,i);
    else
        p=p+1;
        w2_m(:,p)=A(:,i);  
    end
end  

scatter(w1_m(1,:), w1_m(2,:), "*b");
scatter(w2_m(1,:), w2_m(2,:), "*r");

m=0;
n=0;

w1_m_error=[0;0];
w2_m_error=[0;0];
for i=1:k
    if(w1_m(2,i)>limit)
        m=m+1;
        w1_m_error(:,m)=w1_m(:,i);
    end
end

for i=1:p
    if(w2_m(2,i)<limit)
        n=n+1;
        w2_m_error(:,n)=w2_m(:,i);
    end
end

error_percent=((m+n)/(N1+N2))*100;
fprintf("error percentage=: %.3d%%\n",error_percent);

scatter(w1_m_error(1,:), w1_m_error(2,:), "*m");
scatter(w2_m_error(1,:), w2_m_error(2,:), "*g");
legend("w1","w2","w1 error","w2 error",'Location','northwest');

%%b4

figure(5);
title("Bayesian classifier");
hold on;

k=0;
p=0;

S1_inv=pinv(S1);
S2_inv=pinv(S2);
for i=1:(N1+N2)
   res1=-0.5*((A(:,i)-m1')'*S1_inv*(A(:,i)-m1'))+0.5*log(det(S1))+log(P_w1);
   res2=-0.5*((A(:,i)-m2')'*S2_inv*(A(:,i)-m2'))+0.5*log(det(S2))+log(P_w2);
    
     if(res1>res2)
        k=k+1;
        w1_b(:,k)=A(:,i);
    else
        p=p+1;
        w2_b(:,p)=A(:,i);  
    end
end  

scatter(w1_b(1,:), w1_b(2,:), "*b");
scatter(w2_b(1,:), w2_b(2,:), "*r");
legend("w1","w2","Location","northwest");

