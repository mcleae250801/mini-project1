%%%%%%%%%%%%%%%%%%%%%%%%%PART 1%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all; close all;

%"Makes room";
%room dimesions 
X=5;
Y=5;
Z=0;
W = .1; %width of wall

%----------------------left side of the room--------------------------- 
Lside= collisionBox(W,Y,Z);
Lside.Pose = [cos(0) -sin(0)  0   -(W/2);
              sin(0)  cos(0)  0   2.5   ;
              0       0       1   0     ;
              0       0       0   1    ];
%shows left side
show(Lside)
hold on;

%-------------------right side of the room----------------------------
Rside = collisionBox(W,Y,Z);
Rside.Pose = [cos(0) -sin(0)  0   5+(W/2);
              sin(0)  cos(0)  0   2.5    ;
              0       0       1   0      ;
              0       0       0   1     ];
%shows right side
show(Rside)
hold on;
          
%-----------------------top side of the room---------------------------
Top = collisionBox(X,W,Z);
Top.Pose = [cos(0) -sin(0)  0   2.5      ;
            sin(0)  cos(0)  0   5+(W/2)  ;
            0        0      1   0        ;
            0        0      0   1       ];
show(Top)
hold on;

%-----------------------bottom side of the room--------------------------
Bottom = collisionBox(X,W,Z); 
Bottom.Pose =[cos(0) -sin(0)  0  2.5     ;  %x
              sin(0)  cos(0)  0  -(W/2)  ;  %y
              0       0       1  0       ;  %z
              0       0       0  1 ]     ;
show(Bottom)
hold on;

%---------------------------------------------------------------------
"Makes person";
r_p = .2 ; z_p = 0;
%3 meters to the left of the robot
person = collisionCylinder(.2,0);
person.Pose =[cos(0) -sin(0)  0  1     ;  %x
              sin(0)  cos(0)  0  0.5   ;  %y
              0       0       1  0     ;  %z
              0       0       0  1 ]   ;
show(person)
hold on;

"makes shelf";
%/////////////////////////////////////////////////////////////////
%rectangular shelf L=.8m and W=.3m
%3.5 meters north of the person, long side facing east
Ls = .8;
Ws = .3;
shelf = collisionBox(Ws,Ls,0);
shelf.Pose =[cos(0) -sin(0)  0  1     ;  %x
             sin(0)  cos(0)  0  4     ;  %y
             0       0       1  0     ;  %z
             0       0       0  1 ]   ;
show(shelf)
hold on;
%//////////////////////////////////////////////////////////////////
"makes table";
%//////////////////////////////////////////////////////////////////
%table is in center of the room and rotated by 45 degrees
S = 0.5;
table = collisionBox(S,S,0);
table.Pose =[cos(pi/4)  -sin(pi/4)   0  2.5     ;  %x
             sin(pi/4)   cos(pi/4)   0  2.5     ;  %y
             0         0             1  0       ;  %z
             0         0             0  1 ]     ;
show(table)
hold on;

%------------------------------------------------------------------------
"makes robort the robot";
%name the robot robert to not have conflicting names
%4 meters right from the origin and .5 meters up
r_r = .3; z_r = 0;
xr = 4;
yr = 0.5;
zr = -.1;
th0 = pi/2;
robort_0 = collisionCylinder(r_r,z_r);
robort_0.Pose =[cos(th0) -sin(th0)    0    xr     ;  %x
                sin(th0)  cos(th0)    0    yr     ;  %y
                0       0             1    zr     ;  %z
                0       0             0    1 ]    ;
p0 = [4;0.5];  
show(robort_0)
hold on;
        
title('Room')
%creates x and y lims
axis([-.5,5.5,-.5,5.5,-1,1])
hold on
%axix off

"Unit Vectors";
%origin 
quiver(0,0,1,0,'r',"linewidth",4)
quiver(0,0,0,1,'b',"linewidth",4)
%person
quiver(1,.5,.75,0,'r',"linewidth",2)
quiver(1,.5,0,.75,'b',"linewidth",2)
%robot
quiver(4,.5,0,.75,'r',"linewidth",2)
quiver(4,.5,-.75,0,'b',"linewidth",2)
%shelf
quiver(1,4,.75,0,'b',"linewidth",2)
quiver(1,4,0,-.75,'r',"linewidth",2)
%table
quiver(2.5,2.5,.75,.75,'r','linewidth',2)
quiver(2.5,2.5,-.75,.75,'b','linewidth',2)

%%%%%%%%%%%%%%%%%%%%%%%%PART 2%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

"Position M";
%creates a new point inbetween the start,table,and shelf
%gives 0m leeway
xM = 3.2; 
yM = 3; 
zM = -.1;
pM = [xM;yM];
thM = 0;
robort_M = collisionCylinder(r_r,z_r);
%distance away from robot and shelf
robort_M.Pose =[cos(thM) -sin(thM)    0  xM     ;  %x
                sin(thM)  cos(thM)    0  yM     ;  %y
                0          0          1  zM      ;  %z
                0          0          0  1 ]    ;
%shows robort in position M
show(robort_M)
hold on;

"Position A";
%position 
pA = [1.55;4];
d = .1;
xA = (1+r_r+d+(Ws/2));
yA = 4;
zA = -.1;
thA = pi;
robort_A = collisionCylinder(r_r,z_r);
%distance away from shelf
d = .1;
robort_A.Pose =[cos(thA) -sin(thA)    0  xA     ;  %x
                sin(thA)  cos(thA)    0  yA     ;  %y
                0          0          1  zA      ;  %z
                0          0          0  1 ]    ;
%shows robort in position a
show(robort_A)
hold on;

"Postion B";
robort_B = collisionCylinder(r_r,z_r);
%distance away from shelf
pB = [1;.5+r_p+r_r+d];
d = .1;
xB = 1;
yB = (.5+r_p+r_r+d);
zB = -.1;
thB = (3*pi)/2;
robort_B.Pose =[cos(thB) -sin(thB)     0   xB      ;  %x
                sin(thB)  cos(thB)     0   yB      ;  %y     
                0       0              1   zB       ;  %z
                0       0              0   1 ]     ;
%shows robort in position b
show(robort_B)
hold on;

%M
quiver(xM,yM,cos(2.4761),sin(2.4761),'r',"linewidth",2)
quiver(xM,yM,-sin(2.4761),cos(2.4761),'b',"linewidth",2)
%A
quiver(xA,yA,0,-1,'b',"linewidth",2)
quiver(xA,yA,-1,0,'r',"linewidth",2)
%B
quiver(xB,yB,1,0,'b',"linewidth",2)
quiver(xB,yB,0,-1,'r',"linewidth",2)

view(0,90)

%/////////////////////////////////////////////////////////////////////
%__________________generates straight line path________________________
%/////////////////////////////////////////////////////////////////////

%make lambda final the normal scalar distance between the two points
lf0M = norm(pM-p0);
%generate lambda vector
N=20;
%λ vector is a 1xN vector 
%makes a row vector with N colunms from 0-λ final A with equal 0 equal
%spacing
l0M=(0:(lf0M/N):lf0M);
%generate the path based on λ
%uses the p(λ)to plot the (x,y) of the straight line path 
path0M = p0*(1-l0M/lf0M) + pM*l0M/lf0M;
% # of elements in l1
n0M=length(l0M);

%show the straight line path(check if bumbs into tables), code not very
%necessay as it is generated in the next loop all together
for i=1:n0M
    robort_0.Pose(1:2,4) = path0M(:,i);
    collisionstatus = checkCollision(robort_0,table);
    %checkswhere the collision is in the straight line
    if collisionstatus == 1
       position = path0M(:,i);
       break 
    else 
    %    disp("all good")
    end
    %show(robort_0);
end

%gererates a second straight line path
lfMA = norm(pA-pM);
lMA=(0:(lfMA/N):lfMA);
nMA = length(lMA);

%generate the path based on λ
%uses the p(λ)to plot the (x,y) of the straight line path 
pathMA = pM*(1-lMA/lfMA) + pA*lMA/lfMA;

%combines path of 0M and MA to make 0MA
l0MA = [l0M lMA+l0M(end)];
nl0MA = length(l0MA);
path0MA= [path0M pathMA];


% show the straight line path 
for i=1:length(l0MA)
    robort_0.Pose(1:2,4) = path0MA(:,i);
    collisionstatus = checkCollision(robort_0,table);
    %checkswhere the collision is in the straight line
    if collisionstatus == 1
       position = path0MA(:,i);
    %else 
    %    disp("all good")
    end
    %show(robort_0);
end


lfAB = norm(pB-pA);
lAB=(lfAB/N:lfAB/N:lfAB);
nAB=length(lAB);
pathAB = pA*(1-lAB/lfAB) + pB*lAB/lfAB;

l0MAB=[l0MA lAB+l0MA(end)];
nl0MAB=length(l0MAB);
path0MAB=[path0MA pathAB];

%theta
th0 = pi/2;
thA = pi;
thB = (3*pi)/2;

%makes the formula θ=(Δθ)Δλ
%generates N terms between 0M and MA to make the correct indexing
%# of indinces along path between 0MA/AB, subtracts 1 because 
%Matlab starts at 1
Nth0A = (length(pathMA)+length(path0M))-1;
NthAB = length(pathAB)-1;

%Δθ
dtheta0A = (thA-th0);
dthetaAB = (thA-th0);

%makes the Δθ divided taking the end point θ's and dviding by the number of
%instance wanted along path to make even spaces 
dth0A = dtheta0A/Nth0A;
dthAB = dthetaAB/NthAB;

%generates the path of the θ starting at θi and ending with θf with equal
%distancing, makes a row vector of [1xNth_] with spacing of dth_
thpath0A = [th0:dth0A:thA];
thpathAB = [thA:dthAB:thB];
%combines the two paths
thpath0MAB = [thpath0A thpathAB];



%plots the path of the straight lines of the robot
%plots the path θi = θ_0 +(θ_desired-θi)/Δl
%plots the quiver as a function of i along the path 0MAB, takes the cos and
%sin functions at each theta to get a quiver vector
%for e1, cos is the horizontal x component and sin is the vertical component
%for e2, -sin is the horizonatal x component and cos is the vertical 
for i=1:length(l0MAB)
    robort_0.Pose(1:2,4)=path0MAB(:,i);
    collisionstatus = checkCollision(robort_0,shelf);
    e1x = cos(thpath0MAB(i));
    e1y = sin(thpath0MAB(i));
    
    quiver(path0MAB(1,i),path0MAB(2,i),e1x*.75,e1y*.75,'r',"linewidth",1)
        
    e2x = -sin(thpath0MAB(i)); %negative because of the rotation matrix
    e2y = cos(thpath0MAB(i));
    quiver(path0MAB(1,i),path0MAB(2,i),e2x*.75,e2y*.75,'b',"linewidth",1)
    %checkswhere the collision is in the straight line
    if collisionstatus == 1
       position = path0MAB(:,i); 
    end
    show(robort_0); 
end

% lambda vs. position 
figure(6);plot(l0MAB,path0MAB,'linewidth',2);
xlabel('path length (lambda) (m)');
ylabel('x-y robot position (m)');
title('robot position over distance(λ)')
%lambda vs theta
figure(7);plot(l0MAB,thpath0MAB,'linewidth',2)
xlabel('path length (lambda) (m)');
ylabel('angle of orientation (theta)');
title('robot orientation angle θ over distance λ')



%/////////////////////////////////////////////////////////////////////
%_______________spline computation___________________________________
%/////////////////////////////////////////////////////////////////////

n_1p = 20;
n_1a = 8;
n_1s = 50;%60;

%spline of whole section 0MAB
l_s = [l0MAB(n0M-n_1p):(l0MAB(n0M+n_1p)-l0MAB(n0M-n_1p))/n_1s:l0MAB(n0M+n_1p)];
p_s = spline([l0MAB(n0M-n_1p:n0M-n_1p+n_1a) l0MAB(n0M+n_1p-n_1a:n0M+n_1p)],...
    path0MAB(:,[n0M-n_1p:n0M-n_1p+n_1a,n0M+n_1p-n_1a:n0M+n_1p]),l_s);

l3 = [l0MAB(1:n0M-n_1p-1) l_s l0MAB(n0M+n_1p+1:nl0MAB)];
path3 = [path0MAB(:,1:n0M-n_1p-1) p_s path0MAB(:,n0M+n_1p+1:nl0MAB)];

l5 = [l_s lAB+l_s(end)];
p5 = [p_s pathAB];

% show the path interpolation
figure(10);plot(l0MAB,path0MAB,l5,p5,'linewidth',2);
legend('x','y','x with spline','y with spline');
title('robot position vs. path length');
xlabel('lambda (m)');ylabel('x-y position of robot (m)')

%just the spline of the 0MA
l_2s = [l0MA(n0M-n_1p):(l0MA(n0M+n_1p)-l0MA(n0M-n_1p))/n_1s:l0MA(n0M+n_1p)];
p_2s = spline([l0MA(n0M-n_1p:n0M-n_1p+n_1a) l0MA(n0M+n_1p-n_1a:n0M+n_1p)],...
    path0MA(:,[n0M-n_1p:n0M-n_1p+n_1a,n0M+n_1p-n_1a:n0M+n_1p]),l_2s);

l4 = [l0MA(1:n0M-n_1p-1) l_2s l0MA(n0M+n_1p+1:nl0MA)];
path4 = [path0MA(:,1:n0M-n_1p-1) p_2s path0MA(:,n0M+n_1p+1:nl0MA)];

%makes a new figure to plot the spline
hold off
figure(2)
grid on
axis([-.5,5.5,-.5,5.5,-1,1]);hold on
show(Lside);hold on;show(Rside);hold on;show(Top);hold on;show(Bottom); 
hold on;show(shelf);hold on;show(table);hold on;show(person);hold on;

%Does the orientation of the theta
th0 = pi/2;
thA = pi;
thB = (3*pi)/2;
%because the spline combines the two paths of 0M and MA, a new path must be
%generated along path 4 with just the 0MA component
Nths0A = (length(path4))-1;
dths0A = dtheta0A/Nths0A;
thspath0A = [th0:dths0A:thA];
%generates the path of orientation for the AB path, no change with spline
NthsAB = length(pathAB)-1;
dthsAB = dthetaAB/NthsAB;
thspathAB = [thA:dthsAB:thB];
%combines the two  paths
thspath0MAB = [thspath0A thspathAB];

%same rotation matrix as before, plots along path3 with l3 indices
for i=1:length(l3)
    robort_0.Pose(1:2,4)=path3(:,i);
    e1sx = cos(thspath0MAB(i));
    e1sy = sin(thspath0MAB(i));
    quiver(path3(1,i),path3(2,i),e1sx*.75,e1sy*.75,'r',"linewidth",1)%.75 length
    
    e2sx = -sin(thspath0MAB(i));
    e2sy = cos(thspath0MAB(i));
    quiver(path3(1,i),path3(2,i),e2sx*.75,e2sy*.75,'b',"linewidth",1)%.75 length
    show(robort_0);
end
%unsure if needed, just replots path with the l4 index
for i=1:length(l4)
    robort_0.Pose(1:2,4)=path3(:,i);
    show(robort_0);
end


%*********************************************
%index with velocity for sraight path
%*********************************************
%path1 = 0M
%path2 = MA
%path3 = AB
%maximum velocity 
umax = [2;2];wmax=1;
%change
umax_AB = [1.75;1.75]; %the w_AB exceeds 1 rad/s so reduced speed along path
                       %umax for w = wmax = 1 is π/2*w = t = 1.5708s
                       %d = lfAB = 2.8218    umax = 1.796 m/s

%max speed for segment 1
path0M_prime = (pM-p0)/lf0M;
l0Mdotmax = min(abs(umax./path0M_prime));
T0M = lf0M/l0Mdotmax;
%{
thM = 3*pi/2;
theta0Mpath_prime = (thM-th0)/lf0M;
lt0Mdotmax = min(abs(wmax./theta0Mpath_prime));
Tt0M = lf0M/lt0Mdotmax;
%}

%max speed of segment2
pathMA_prime = (pA-pM)/lfMA;
lMAdotmax=min(abs(umax./pathMA_prime));
TMA = lfMA/lMAdotmax;

thetaMApath_prime = (thA-thM)/lfMA;
ltMAdotmax = min(abs(wmax./thetaMApath_prime));

%max speed of segment3
pathAB_prime = (pB-pA)/lfAB;
%lABdotmax=min(abs(umax./pathAB_prime));
lABdotmax=min(abs(umax_AB./pathAB_prime));
TAB = lfAB/lABdotmax;

thetaABpath_prime = (thB-thA)/lfAB;
ltABdotmax = min(abs(wmax./thetaABpath_prime));

%use vehicle kinematcs to generate motion
ts=.1; % 0.1 second sampling period
t0M=(0:ts:T0M);NT0M=length(t0M);
pt0M=zeros(2,NT0M);pt0M(:,1)=p0;
% use simple finite difference to propagate robot motion
% assuming you can control the robot velocity
for i=1:NT0M-1
    pt0M(:,i+1)=pt0M(:,i)+ts*owmr(pt0M(:,i),path0M_prime*l0Mdotmax,umax,-umax);
    
end

% second segment
tMA=(T0M:ts:T0M+TMA);NTMA=length(tMA);
ptMA=zeros(2,NTMA);ptMA(:,1)=pt0M(:,NT0M);
for i=1:NTMA-1
    ptMA(:,i+1)=ptMA(:,i)+ts*owmr(ptMA(:,i),pathMA_prime*lMAdotmax,umax,-umax);
end

%third segment 
tAB=(TMA+T0M:ts:TAB+TMA+T0M);NTAB=length(tAB);
ptAB=zeros(2,NTAB);ptAB(:,1)=ptMA(:,NTMA);
for i=1:NTAB-1
    %ptAB(:,i+1)=ptAB(:,i)+ts*owmr(ptAB(:,i),pathAB_prime*lABdotmax,umax,-umax);
    ptAB(:,i+1)=ptAB(:,i)+ts*owmr(ptAB(:,i),pathAB_prime*lABdotmax,umax_AB,-umax_AB);
end

% combine motion between first, second, and third segment
t1=[t0M tMA(2:NTMA) tAB(2:NTAB)];
pt1=[pt0M ptMA(:,2:NTMA) ptAB(:,2:NTAB)];

% calculate path length
lt1=[0 cumsum(vecnorm(diff(pt1')'))];

%_____________________theta orientation__________________________
%calculates the theta position based on time a angular velocity
%uses the time from the maximum velocity to get a minimum time to do the
%rotations of each segment

Time_0M = t0M(end) ;%time it takes to got from 0 to M
Time_MA = tMA(end)-Time_0M; %time it takes to go from M to A
Time_AB = tAB(end)-Time_MA-Time_0M; %time it takes to go from A to B


lfOMA = lf0M+lfMA; %length of the full segment bwteen 0MA

%find the orientation of M based on the lambda length and difference in 
%theta. Makes the theta M proportion to the distance to M 
%uses formula: 
%distance traveled between 0M/distance traveled between 0MA = θ_M/Δθ
thM = ((lf0M * (thA-th0)) /(lf0M + lfMA))+(pi/2);
%finds the actual w based on time 
w_0M = (thM-th0)/(Time_0M);
w_MA = (thA-thM)/(Time_MA);
w_AB = (thB-thA)/(Time_AB);

%w_AB is always larger tan wmax so switches the time to be longer
%robot will continue to rotate after moving in x-y direction to account for
%the wmax being too slow in the time it takes robto to go from A to B at
%max velocity
if w_AB > wmax
    deltaT_AB = (thB-thA);
    Time_AB = deltaT_AB;
    disp("w_AB is larger than w_max")
else
    Time_AB=tAB(end)-Time_MA-Time_0M;
    disp("all good")
end

%picks to have 10 columns per path
N2 = 10;
path_theta_0M = [linspace(th0,thM,N2);
                 linspace(0,Time_0M,N2)];
path_theta_MA = [linspace(thM,thA,N2);
                 linspace(Time_0M,(Time_0M+Time_MA),N2)];
path_theta_AB = [linspace(thA,thB,N2);
                 linspace((Time_0M+Time_MA),(Time_0M+Time_MA+Time_AB),N2)];
%combines all three paths
path_theta_0MA = [path_theta_0M path_theta_MA];
path_theta_0MAB = [path_theta_0MA path_theta_AB];


% motion vs. time 
figure(11);plot(t1,pt1,'linewidth',2);legend('x','y');
xlabel('time (sec)');ylabel('robot position (m)');
title('motion vs time');
% lambda vs. time (piecewise linear, with a kink at transition)
figure(12);plot(t1,lt1,'linewidth',2);
xlabel('time (sec)');ylabel('path length (lambda) (m)');
title('path velocity for straight path')
figure(3);plot(path_theta_0MAB(2,:),path_theta_0MAB(1,:),'linewidth',2);
xlabel('time (sec)');ylabel('angle of orientaion (theta)');
title('angle of orientation vs time')


%***********************************
% index path with time (with spline)
%***********************************
% first compute p' in the spline region
umax2 = .2;
ps_prime=diff(p_s')'./diff(l_s);
% find the maximum path velocity that won't violate the velocity constraint
lsdotmax=min(min(abs(umax2./ps_prime)')');
% use a single constant velocity for the entire path 
ldotmax=min([l0Mdotmax lMAdotmax lsdotmax]);

% calculate the time needed to travel the complete path
T3=l3(end)/ldotmax; % faster because the path length is shorter
% calculate p' for the entire path
p3_prime=diff(path3')'./diff(l3);
% add one more at the end because taking the difference loses one element
p3_prime=[p3_prime p3_prime(:,end)];
% set up time vector at regular sampling period
t3=(0:ts:T3);NT3=length(t3);
% set up storage space
pt3=zeros(2,NT3);pt3(:,1)=p0;
lt3=zeros(size(t3));
pt3prime=zeros(2,NT3);
% the initial path slope is just from the path itself
pt3prime(:,1)=(path3(:,2)-path3(:,2))/norm(path3(:,2)-path3(:,1));
% use robot kinematics to propagate, and the approximate p' and the maximum
% path velocity to set the robot velocity
u3=zeros(2,NT3-1);
for i=1:NT3-1
    u3(:,i)=pt3prime(:,i)*ldotmax;
    pt3(:,i+1)=pt3(:,i)+ts*owmr(pt3(:,i),u3(:,i),umax,-umax);
    lt3(i+1)=lt3(i)+ts*ldotmax;
    xprime=interp1(l3(1:end),p3_prime(1,:),lt3(i+1));
    yprime=interp1(l3(1:end),p3_prime(2,:),lt3(i+1));
    pt3prime(:,i+1)=[xprime;yprime];
end
% compare the ideal path vs. the approximate path using robot kinematics
figure(13);plot(lt3,pt3,l3,path3,'x','linewidth',2);
legend('x kinematics','y kinematics','x path','y path');
xlabel('lambda (m)');xlabel('robot position (m)');
title('comparison between robot motion based on kinematics model and geometric path');

figure(14);plot(t3,pt3,'linewidth',2);legend('x','y');
xlabel('time (sec)');ylabel('robot position (m)');
title('robot motion based on kinematics model');

figure(15);plot(t3(1:end-1),u3,'linewidth',2);legend('u_x','u_y');
xlabel('time (sec)');ylabel('robot position (m)');
title('robot velocity')

%
% constant path speed: ldotmax
% path with spline: l3 and path3 
%

% calculate corresponding time
t3a=l3/ldotmax;
% calculate speed for each segment
u3a=diff(path3')'./diff(t3a);
%
figure(24);plot(t3a,path3,'linewidth',2);legend('x','y');
xlabel('time (sec)');ylabel('robot position (m)');
title('robot motion based on kinematics model');

figure(25);plot(t3a(1:end-1),u3a,'linewidth',2);legend('u_x','u_y');
xlabel('time (sec)');ylabel('robot position (m)');
title('robot velocity')


function Xdot = owmr(XD,U,Umax,Umin)
    Xdot = max(min(U,Umax),Umin);
end



%
% kinematics of an nonholonomic wheeled mobile robot
%
% input: 
%       X = current state (x, y, theta)
%       U = current input (u=forward velocity, w=turning angular velocity)
% 
% output: 
%       Xdot = current velocity
%
function Xdot = nwmr(X,U,Umax,Umin)

    Xdot = [cos(X(3)) 0;sin(X(3)) 0; 0 1]*max(min(U,Umax),Umin);
    
end