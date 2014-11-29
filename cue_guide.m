function cue_guide()
% Shaun Weatherford, June 2013
% from picture of pool table with balls and pockets, rectifies the image to
% create a top-down view, identifies table walls, pockets, stripe and solid
% balls, and cue and eight balls, then adds cue guide vectors to image
close all;clear all

% determine whether to plot intermediate images
plot_binarize=0;
plot_erode=0;
plot_pocket=0;
plot_rectified=0;
plot_mark_template=0;

% insert name of pool table picture here

picture=4;
view_top=0;
hires=1;

if picture==1
    picture=imread('photo.JPG');
elseif picture==2
    picture=imread('photo2_angle_sunlight.JPG');
elseif picture==3
    picture=imread('photo3_angle_lamp.JPG');
elseif picture==4
    if view_top==1
        if hires==1
            picture=imread('photo4_top_sun.JPG');
        else
            picture=imread('photo4_top_sun_lores.JPG');
        end
    else
        if hires==1
            picture=imread('photo4_front_sun.JPG');
        else
            picture=imread('photo4_front_sun_lores.JPG');
        end
    end
end





picture_double=im2double(picture);

height=size(picture_double,1);
width=size(picture_double,2);
corner_upper_left=zeros(2,1);
corner_upper_right=zeros(2,1);
corner_lower_left=zeros(2,1);
corner_lower_right=zeros(2,1);

figure;
imshow(picture_double);
t = text(round(width/100),round(height/15),'Original Picture','FontSize',20, 'FontWeight','demi');

% gets red, green and blue components of big picture
picture_r=picture_double(:,:,1);
picture_g=picture_double(:,:,2);
picture_b=picture_double(:,:,3);
picture_mag=(picture_r.^2+picture_g.^2+picture_b.^2).^0.5;
picture_r_unit=picture_r./picture_mag;
picture_g_unit=picture_g./picture_mag;
picture_b_unit=picture_b./picture_mag;
figure;
imshow(picture_mag);
figure;
imshow(picture_r_unit);
figure;
imshow(picture_b_unit);
figure;
imshow(picture_g_unit);

%{
finds table color
[location_r,location_g,location_b,location_m]=find_table_color(height,width,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit);

% isolates table and pockets, via inner product of color vector with
% picture, binarize image, erode object to reduce noise, then region labeling of
% pockets
otsu_flag=0;
% Binarize Image
[picture_delta,picture_filter,cent,bound,picture_mag]=binarize_image(picture_double,location_r,location_g,location_b,location_m,otsu_flag,plot_binarize,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right);



% Rectify Image such that table edge and wall lines are parallel to picture
% x and y axes (clean top-down view); will help in accurately simulating
% ball impacts and bounces from walls; rectifies by identifying outer edge
% corners, via projection of outer edge wall lines from above, the using
% homography to find transformation matrix such that picture corners match
% closely with reference image corners
[corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right] = find_corners(picture_filter);
[picture_double]=rectify_image(picture_filter,picture_double,plot_rectified,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right);



% gets red, green and blue components of big picture
picture_r=picture_double(:,:,1);
picture_g=picture_double(:,:,2);
picture_b=picture_double(:,:,3);
picture_mag=(picture_r.^2+picture_g.^2+picture_b.^2).^0.5;
picture_r_unit=picture_r./picture_mag;
picture_g_unit=picture_g./picture_mag;
picture_b_unit=picture_b./picture_mag;





picture_marked=picture_double;

% Find Balls (precise localization but misses green): repeats above inner product of color
% table with rectified picture, binarizes, then XORs with a boundingbox filled image,
% to leave mostly balls and pockets binarized white, then erodes to reduce
% noise and separate balls and pockets from each other, region labels pockets and balls
otsu_flag=0;
[picture_delta,picture_filter,cent,bound,picture_mag]=binarize_image(picture_double,location_r,location_g,location_b,location_m,otsu_flag,plot_binarize,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right);
[SEradius,SE,picture_label,area,centroid,boundingbox,picture_dilate]=erode_objects(bound,picture_delta,plot_erode,70);
[pocket,centroid_pocket,boundingbox_pocket,pocketradius,picture_pockets,NumberPockets]=find_pockets(SE,picture_label,cent,bound,area,centroid,boundingbox,SEradius,plot_pocket);
[NumberBalls,ball, centroid_ball,boundingbox_ball,ballradius]=find_balls(pocketradius,centroid_pocket,picture_delta,picture_double,area,pocket,centroid,boundingbox,SEradius,0.9);
NumberBallsOrig=NumberBalls;
% Find Balls (less precise localization but finds green): previous approach
% inadequate for green balls (removes green balls); uses Otsu threshold to include
% non-table shades of green in ball labeling - but susceptible to ball
% shadows on table; ball find uses AND convolution of binarized ball images
% with averaged then binarized ball template to more accurately determine
% each ball's location
otsu_flag=1;
[picture_delta2,picture_filter2,cent2,bound2,picture_mag2]=binarize_image(picture_double,location_r,location_g,location_b,location_m,otsu_flag,plot_binarize,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right);
[SEradius,SE,picture_label2,area2,centroid2,boundingbox2,picture_dilate2]=erode_objects(bound,picture_delta2,plot_erode,60);
[NumberBalls2,ball2, centroid_ball2,boundingbox_ball2,ballradius2]=find_balls(pocketradius,centroid_pocket,picture_delta,picture_double,area2,pocket,centroid2,boundingbox2,SEradius,0.8);

% Merge Green balls to the ball list
[ball,centroid_ball,boundingbox_ball,NumberBalls]=merge_greens(NumberBalls,NumberBalls2,ball,centroid_ball,boundingbox_ball,ball2,centroid_ball2,boundingbox_ball2,ballradius);

% Find Cue and Eight balls, then Find Stripes and Solids; eight and cue
% are found mostly from amplitude (gray space) and RGB spectrum in white
% region; also white balances balls from cue ball spectrum to help
% determine accurate percentage of white for each ball, to separate solids
% from stripes

[cue_ball,centroid_cue,multmat,NumberEight,eight_ball,centroid_eight]=find_cue_eight(NumberBallsOrig,centroid_ball,ballradius,picture_delta,picture_double);
[picture_marked,solid_ball,centroid_solid,stripe_ball,centroid_stripe,NumberSolids,NumberStripes]=find_stripes_solids(cue_ball,eight_ball,multmat,NumberBalls,centroid_ball,ballradius,picture_delta,picture_double,picture_marked);

% Find Walls in rectified image, to be used for simulation of reflections
% (taking place of ball bounces) for cue guide
[slope,intercept,endp7,endp8,endp9,endp10,endp11,endp12]=find_walls(centroid_pocket,bound,picture_filter,picture_mag,picture_pockets);

% Mark Image to label solids, stripes, cue, eight,
% walls and pockets
[picture_marked]=mark_image(picture_marked,ballradius,NumberSolids,centroid_solid,NumberStripes,centroid_stripe,centroid_cue,centroid_eight,centroid_pocket);

hold on;
plot(endp7(:,1),endp7(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp8(:,1),endp8(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp9(:,1),endp9(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp10(:,1),endp10(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp11(:,1),endp11(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp12(:,1),endp12(:,2),'LineWidth',2,'Color','cyan');

% Find Reflections of balls and pockets, instead of simulating bounces off of table walls
[NumberStripesTmp,centroid_stripe,NumberSolidsTmp,centroid_solid,NumberEightTmp,centroid_eight,NumberPocketsTmp,centroid_pocket,rotation_angle_top,rotation_point_top,rotation_angle_bottom,rotation_point_bottom,rotation_angle_left,rotation_point_left,rotation_angle_right,rotation_point_right]=find_reflections(ballradius,slope,intercept,NumberStripes,centroid_stripe,NumberSolids,centroid_solid,NumberEight,centroid_eight,NumberPockets,centroid_pocket);

% Find Ball Angles for use in cue guide calculations
[dist_cue_stripe,angle_cue_stripe,angledel_cue_stripe,dist_cue_solid,angle_cue_solid,angledel_cue_solid,dist_cue_eight,angle_cue_eight,angledel_cue_eight]=find_ball_angles(ballradius,centroid_cue,NumberStripesTmp,centroid_stripe,NumberSolidsTmp,centroid_solid,NumberEightTmp,centroid_eight);

if plot_mark_template==1
    figure;
    imshow(picture_marked);
end

hold on;
plot(endp7(:,1),endp7(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp8(:,1),endp8(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp9(:,1),endp9(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp10(:,1),endp10(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp11(:,1),endp11(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp12(:,1),endp12(:,2),'LineWidth',2,'Color','cyan');

% Find Cue Guides for stripes
[junk] = find_cue_guides(slope,intercept,centroid_cue,ballradius,pocketradius,NumberStripes,NumberStripesTmp,angle_cue_stripe,angledel_cue_stripe,dist_cue_stripe,centroid_stripe,NumberSolids,NumberSolidsTmp,centroid_solid,NumberEight,NumberEightTmp,centroid_eight,NumberPockets,NumberPocketsTmp,centroid_pocket,0,rotation_angle_top,rotation_angle_bottom,rotation_angle_left,rotation_angle_right,rotation_point_top,rotation_point_bottom,rotation_point_left,rotation_point_right);
height=size(picture_double,1);
width=size(picture_double,2);

t = text(round(width/100),round(height/15),'Stripe Cue Guides','FontSize',20, 'FontWeight','demi');

figure;
imshow(picture_marked);


hold on;
plot(endp7(:,1),endp7(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp8(:,1),endp8(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp9(:,1),endp9(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp10(:,1),endp10(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp11(:,1),endp11(:,2),'LineWidth',2,'Color','cyan');
hold on;
plot(endp12(:,1),endp12(:,2),'LineWidth',2,'Color','cyan');

% Find Cue Guides for solids
[junk] = find_cue_guides(slope,intercept,centroid_cue,ballradius,pocketradius,NumberSolids,NumberSolidsTmp,angle_cue_solid,angledel_cue_solid,dist_cue_solid,centroid_solid,NumberStripes,NumberStripesTmp,centroid_stripe,NumberEight,NumberEightTmp,centroid_eight,NumberPockets,NumberPocketsTmp,centroid_pocket,0,rotation_angle_top,rotation_angle_bottom,rotation_angle_left,rotation_angle_right,rotation_point_top,rotation_point_bottom,rotation_point_left,rotation_point_right);
t = text(round(width/100),round(height/15),'Solid Cue Guides','FontSize',20, 'FontWeight','demi');



end



















function [centroid_object,NumberObjectsTmp] = reflections(centroid_object,NumberObjects,NumberObjectsTmp,rotation_angle,rotation_point,reflectx,addradius,ballradius)
% rotates objects and walls such that walls are either perfectly vertical
% or horizontal, then reflects objects from walls, then rotates back
% image rectification should make this unnecessary, but done just in case
rotation_matrix=[cos(rotation_angle),sin(rotation_angle);
    -sin(rotation_angle),cos(rotation_angle)];
invrotation_matrix=[cos(-rotation_angle),sin(-rotation_angle);
    -sin(-rotation_angle),cos(-rotation_angle)];
for object=1:NumberObjects
    % rotation
    centroidtmp=rotation_matrix*centroid_object(object,:)';
    
    % reflection
    if reflectx==1
        if addradius==1
            centroidtmp2(1)=2*rotation_point(1)-centroidtmp(1)+2*ballradius;
        else
            centroidtmp2(1)=2*rotation_point(1)-centroidtmp(1)-2*ballradius;
        end
        centroidtmp2(2)=centroidtmp(2);
    else
        centroidtmp2(1)=centroidtmp(1);
        if addradius==1
            centroidtmp2(2)=2*rotation_point(2)-centroidtmp(2)+2*ballradius;
        else
            centroidtmp2(2)=2*rotation_point(2)-centroidtmp(2)-2*ballradius;
        end
    end
    
    %rotation back
    centroid_object(object+NumberObjectsTmp,:)=invrotation_matrix*centroidtmp2';
end
NumberObjectsTmp=size(centroid_object,1);

end

function [centroid_object] = unreflections(centroid_object,rotation_angle,rotation_point,reflectx,addradius,ballradius)
% executes exact same algorithm as "reflections", but does so one obect at
% a time, and does so on a reflection to find corresponding table
% coordinates
rotation_matrix=[cos(rotation_angle),sin(rotation_angle);
    -sin(rotation_angle),cos(rotation_angle)];
invrotation_matrix=[cos(-rotation_angle),sin(-rotation_angle);
    -sin(-rotation_angle),cos(-rotation_angle)];

% rotation
centroidtmp=rotation_matrix*centroid_object';

% "un"reflection
if reflectx==1
    if addradius==1
        centroidtmp2(1)=2*rotation_point(1)-centroidtmp(1)+2*ballradius;
    else
        centroidtmp2(1)=2*rotation_point(1)-centroidtmp(1)-2*ballradius;
    end
    centroidtmp2(2)=centroidtmp(2);
else
    centroidtmp2(1)=centroidtmp(1);
    if addradius==1
        centroidtmp2(2)=2*rotation_point(2)-centroidtmp(2)+2*ballradius;
    else
        centroidtmp2(2)=2*rotation_point(2)-centroidtmp(2)-2*ballradius;
    end
end

% rotation back
centroid_object=invrotation_matrix*centroidtmp2';

end

function [corner]=cornerfind(intercept1,intercept2,slope1,slope2)
% finds projections of outer table edge lines to the corners; corners are
% good markers for use by the rectification algorithm
corner(1)=(intercept2-intercept1)./(slope1-slope2);
corner(2)=slope1.*corner(1)+intercept1;
end


function [slope,slope2,intercept,intercept2,endp,buffer] = wallfind(dely,cent1,cent2,ratio,picture_filter,picture_mag,picture_pockets,buffer,clear_buffer,addmult,usex,startend)
% finds outer table edges and inner bounce walls by radiating inward from
% edges of binarized table picture until non-zero (white) binarized table
% edges found;  uses Hough Transform to find corresponding line
% characteristics; from located edges of table, creates gradient vector and
% integrates vectors along table edge to find inner bounce wall offset from
% outer table edge (under assumption inner wall is parallel to table edge and
% integration may be needed if inner bounce walls have low gradient
% magnitudes and may be hidden by noise
start=round(cent1+1/ratio*(cent2-cent1));
stop=round(cent1+(ratio-1)/ratio*(cent2-cent1));

picture_zero=picture_mag.*0;

index=1;
if clear_buffer==1
    buffer=zeros(dely,1);
end
if usex==1
    % finding top or bottom table edges
    for y=start:stop
        if startend==1
            x=size(picture_filter,2);
        else
            x=1;
        end
        % looks for binarized white edges of table
        while (picture_filter(y,x)==0)
            x=x+addmult.*1;
        end
        % saves edges in arrays
        xarray(index)=x;
        yarray(index)=y;
        picture_zero(y,x)=1;
        x=x+addmult.*2;
        % integration of gradient vector
        buffer=buffer+(picture_mag(y,x+addmult:addmult:x+addmult.*dely)-picture_mag(y,x-addmult:addmult:x+addmult.*dely-addmult.*2))';
        index=index+1;
    end
else
    % finding left or right table edges
    for x=start:stop
        if startend==1
            y=size(picture_filter,1);
        else
            y=1;
        end
        while (picture_filter(y,x)==0)
            y=y+addmult.*1;
        end
        xarray(index)=x;
        yarray(index)=y;
        picture_zero(y,x)=1;
        y=y+addmult.*2;
        buffer=buffer+(picture_mag(y+addmult:addmult:y+addmult.*dely,x)-picture_mag(y-addmult:addmult:y+addmult.*dely-addmult.*2,x));
        index=index+1;
    end
end
% Hough transform
[H,theta,rho]=hough(picture_zero,'Theta',-90:0.5:89.5);
peaks=houghpeaks(H,1);
t=theta(peaks(1,2))*pi/180;
r=rho(peaks(1,1));
if tan(t)==0
    slope=99999;
else
    slope=-1./tan(t);
end
if sin(t)==0
    intercept=-r*slope;
else
    intercept=r./sin(t);
end

% finds line characteristics of inner bounce wall, given offset distance from outer
% edge
buffer2=buffer(3:length(buffer));
[maxval,maxloc]=max((buffer2));
[minval,minloc]=min((buffer2));
intercept_offset=min(maxloc,minloc);
if usex==0
    intercept2=intercept+addmult.*intercept_offset;
else
    intercept2=intercept-addmult.*slope.*intercept_offset;
end
slope2=slope;

% traverses length of found inner wall lines until pockets found in
% binarized pocket image (finds white pixels); creates line start and stop
% points for each wall
if usex==1
    yinit=start;
    xinit=(yinit-intercept2)./slope2;
    y=yinit;
    x=xinit;
    while(picture_pockets(round(y),round(x))==0)
        y=y-1;
        x=(y-intercept2)./slope2;
    end
    endp(1,1)=x;
    endp(1,2)=y;
    y=yinit;
    x=xinit;
    while(picture_pockets(round(y),round(x))==0)
        y=y+1;
        x=(y-intercept2)./slope2;
    end
    endp(2,1)=x;
    endp(2,2)=y;
else
    xinit=start;
    yinit=slope2.*xinit+intercept2;
    x=xinit;
    y=yinit;
    while(picture_pockets(round(y),round(x))==0)
        x=x-1;
        y=slope2.*x+intercept2;
    end
    endp(1,1)=x;
    endp(1,2)=y;
    x=xinit;
    y=yinit;
    while(picture_pockets(round(y),round(x))==0)
        x=x+1;
        y=slope2.*x+intercept2;
    end
    endp(2,1)=x;
    endp(2,2)=y;
end

end

function [junk] = find_cue_guides(slope,intercept,centroid_cue,ballradius,pocketradius,NumberStripes,NumberStripesTmp,angle_cue_stripe,angledel_cue_stripe,dist_cue_stripe,centroid_stripe,NumberSolids,NumberSolidsTmp,centroid_solid,NumberEight,NumberEightTmp,centroid_eight,NumberPockets,NumberPocketsTmp,centroid_pocket,dottedline,rotation_angle_top,rotation_angle_bottom,rotation_angle_left,rotation_angle_right,rotation_point_top,rotation_point_bottom,rotation_point_left,rotation_point_right)
top_edge=(intercept(7)+intercept(8))/2+ballradius;
bottom_edge=(intercept(9)+intercept(10))/2-ballradius;
left_edge=-intercept(11)./slope(11)+ballradius;
right_edge=-intercept(12)./slope(12)-ballradius;
% for given ball type (stripe or solid), cycles through balls and their
% reflected images, and for each ball, simulates cue hits at angles from
% one edge of the ball to the opposite edge of the ball, in one pixel
% increments; then from the ball hit angle, simulates the travel direction
% of the ball and checks whether lands in a pocket; if does, collects all
% cue angles which resulted in lands and culls out the two with smallest
% ball hit angle from center, and with smallest distance to landed pocket
for object=1:NumberStripesTmp
    indexstart=angle_cue_stripe(object)-angledel_cue_stripe(object);
    indexstop=angle_cue_stripe(object)+angledel_cue_stripe(object);
    stepsize=180/pi*atan(1/40*ballradius./dist_cue_stripe(object));
    angleindex=1;
    % cycles through different ball impact angles, from one edge to the
    % opposite
    for angle=indexstart:stepsize:indexstop
        success_flag(angleindex)=0;
        hit_angle(angleindex)=0;
        hit_point(angleindex,1:2)=0;
        pocket_num(angleindex)=0;
        pocket_dist(angleindex)=0;
        stop_flag=0;
        travel_pos(1)=centroid_cue(1,1);
        travel_pos(2)=centroid_cue(1,2);
        travel_dist=0;
        dist_tgol=99999999;
        dist_tgo=999999;
        hit_vector(1)=cos(angle*pi/180);
        hit_vector(2)=sin(angle*pi/180);
        % makes sure the cue ball is still approaching the impact ball;
        % once the cue ball passes the ball, or is within a ball diameter,
        % or impacts another ball along the way (causing stop flag=1), then
        % stops cue ball to ball travel calcs
        while(dist_tgo<dist_tgol && dist_tgo>2*ballradius && stop_flag==0)
            % checks against impacts with other balls in the same region
            % (for cue in reflection region, checks impacts only against
            % other balls in same reflection region
            if travel_pos(1)<left_edge
                startmultiplier=3;
            elseif travel_pos(1)>right_edge
                startmultiplier=4;
            elseif travel_pos(2)<top_edge
                startmultiplier=1;
            elseif travel_pos(2)>bottom_edge
                startmultiplier=2;
            else
                startmultiplier=0;
            end
            endmultiplier=startmultiplier+1;
            startmultiplier=0;
            endmultiplier=5;
            minobdist=999999;
            % checks against impacts with other balls before hitting ball
            % of interest; finds minimum distance to other balls and
            % objects, so can integrate sim at faster rate
            for object2=startmultiplier*NumberStripes+1:endmultiplier*NumberStripes
                if object~=object2
                    dist=sqrt((travel_pos(1)-centroid_stripe(object2,1)).^2+(travel_pos(2)-centroid_stripe(object2,2)).^2);
                    if dist<=2.2*ballradius
                        stop_flag=1;
                    end
                    if dist<minobdist
                        minobdist=dist;
                    end
                end
            end
            for object2=startmultiplier*NumberSolids+1:endmultiplier*NumberSolids
                dist=sqrt((travel_pos(1)-centroid_solid(object2,1)).^2+(travel_pos(2)-centroid_solid(object2,2)).^2);
                if dist<=2.2*ballradius
                    stop_flag=1;
                end
                if dist<minobdist
                    minobdist=dist;
                end
            end
            for object2=startmultiplier*NumberEight+1:endmultiplier*NumberEight
                dist=sqrt((travel_pos(1)-centroid_eight(object2,1)).^2+(travel_pos(2)-centroid_eight(object2,2)).^2);
                if dist<=2.2*ballradius
                    stop_flag=1;
                end
                if dist<minobdist
                    minobdist=dist;
                end
            end
            for object2=startmultiplier*NumberPockets+1:endmultiplier*NumberPockets
                dist=sqrt((travel_pos(1)-centroid_pocket(object2,1)).^2+(travel_pos(2)-centroid_pocket(object2,2)).^2);
                if dist<=pocketradius
                    stop_flag=1;
                end
                if dist<minobdist
                    minobdist=dist;
                end
            end
            for wallindex=7:12
                dist=abs((travel_pos(2)-slope(wallindex).*travel_pos(1)-intercept(wallindex))./(slope(wallindex).*hit_vector(1)-hit_vector(2)));
                if dist<minobdist
                    minobdist=dist;
                end
            end
            dist=sqrt((travel_pos(1)-centroid_stripe(object,1)).^2+(travel_pos(2)-centroid_stripe(object,2)).^2);
            if dist<minobdist
                minobdist=dist;
            end
            minobdist=minobdist-2*ballradius;
            dist_tgol=dist_tgo;
            dist_tgo=sqrt((travel_pos(1)-centroid_stripe(object,1)).^2+(travel_pos(2)-centroid_stripe(object,2)).^2);
            % speeds up integration rate
            travel_pos=travel_pos+max(1.0,0.95*minobdist).*hit_vector;
            % if cue ball hits table wall before impacting reflected ball, notes
            % table impact point
            if travel_pos(2)>top_edge && travel_pos(2)<bottom_edge && travel_pos(1)>left_edge && travel_pos(1)<right_edge
                hitb(1:2)=travel_pos(1:2);
            end
            hit(1:2)=travel_pos(1:2);
        end
        if stop_flag==0
            % if cue ball hit impact ball, now checks whether impact ball
            % lands in pocket
            hit_vector(1:2)=centroid_stripe(object,1:2)-travel_pos(1:2);
            hit_vector_mag=sqrt(hit_vector(1).^2+hit_vector(2).^2);
            hit_vector=hit_vector./hit_vector_mag;
            travel_pos(:)=centroid_stripe(object,:);
            maxdist=0;
            for object3=1:NumberPocketsTmp
                dist=sqrt((centroid_stripe(object,1)-centroid_pocket(object3,1)).^2+(centroid_stripe(object,2)-centroid_pocket(object3,2)).^2);
                if dist>maxdist
                    maxdist=dist;
                end
            end
            travel_dist=0;
            dist_tgol=99999999;
            dist_tgo=999999;
            distmin=9999999;
            while(travel_dist<maxdist && stop_flag==0)
                if travel_pos(1)<left_edge
                    startmultiplier=3;
                elseif travel_pos(1)>right_edge
                    startmultiplier=4;
                elseif travel_pos(2)<top_edge
                    startmultiplier=1;
                elseif travel_pos(2)>bottom_edge
                    startmultiplier=2;
                else
                    startmultiplier=0;
                end
                endmultiplier=startmultiplier+1;
                startmultiplier=0;
                endmultiplier=5;
                minobdist=999999;
                % again, checks for impacts with other balls before landing
                % in pocket
                if object>NumberStripes
                    if travel_pos(2)>top_edge && travel_pos(2)<bottom_edge && travel_pos(1)>left_edge && travel_pos(1)<right_edge
                        stop_flag=1;
                    end
                    flrtmp=floor((object-1)./NumberStripes);
                    if flrtmp<3
                        if travel_pos(1)<left_edge-2*ballradius || travel_pos(1)>right_edge+2*ballradius
                            stop_flag=1;
                        end
                    else
                        if travel_pos(2)<top_edge-2*ballradius || travel_pos(2)>bottom_edge+2*ballradius
                            stop_flag=1;
                        end
                    end
                    
                end
                for object2=startmultiplier*NumberStripes+1:endmultiplier*NumberStripes
                    if object~=object2
                        dist=sqrt((travel_pos(1)-centroid_stripe(object2,1)).^2+(travel_pos(2)-centroid_stripe(object2,2)).^2);
                        if dist<=2.5*ballradius
                            stop_flag=1;
                        end
                        if dist<minobdist
                            minobdist=dist;
                        end
                    end
                end
                for object2=startmultiplier*NumberSolids+1:endmultiplier*NumberSolids
                    dist=sqrt((travel_pos(1)-centroid_solid(object2,1)).^2+(travel_pos(2)-centroid_solid(object2,2)).^2);
                    if dist<=2.5*ballradius
                        stop_flag=1;
                    end
                    if dist<minobdist
                        minobdist=dist;
                    end
                end
                for object2=1:startmultiplier*NumberEight+1:endmultiplier*NumberEight
                    dist=sqrt((travel_pos(1)-centroid_eight(object2,1)).^2+(travel_pos(2)-centroid_eight(object2,2)).^2);
                    if dist<=2.5*ballradius
                        stop_flag=1;
                    end
                    if dist<minobdist
                        minobdist=dist;
                    end
                end
                travel_dist=travel_dist+sqrt(hit_vector(1).^2+hit_vector(2).^2);
                if travel_pos(2)>top_edge && travel_pos(2)<bottom_edge && travel_pos(1)>left_edge && travel_pos(1)<right_edge
                    hit2b(1:2)=travel_pos(1:2);
                end
                hit2(1:2)=travel_pos(1:2);
                if stop_flag==0
                    mindist=999999;
                    for object3=1:NumberPocketsTmp
                        if stop_flag==0
                            dist=sqrt((travel_pos(1)-centroid_pocket(object3,1)).^2+(travel_pos(2)-centroid_pocket(object3,2)).^2);
                            if dist<minobdist
                                minobdist=dist;
                            end
                            if dist<mindist
                                mindist=dist;
                            end
                            % if impact ball within critical distance of
                            % pocket, declares success and saves
                            % information for later culling of "best" cue
                            % guides
                            if dist<0.8*pocketradius-ballradius
                                stop_flag=1;
                                success_flag(angleindex)=1;
                                hit_angle(angleindex)=angle;
                                hit_pointb(angleindex,:)=hitb(:);
                                hit_point(angleindex,:)=hit(:);
                                hit_point2b(angleindex,:)=hit2b(:);
                                hit_point2(angleindex,:)=hit2(:);
                                pocket_num(angleindex)=object3;
                                pocket_dist(angleindex)=sqrt((centroid_stripe(object,1)-centroid_pocket(object3,1)).^2+(centroid_stripe(object,2)-centroid_pocket(object3,2)).^2);
                                pocket_dist2(angleindex)=sqrt((hit2(1)-centroid_pocket(object3,1)).^2+(hit2(2)-centroid_pocket(object3,2)).^2);
                            end
                        end
                    end
                end
                for wallindex=7:12
                    dist=abs((travel_pos(2)-slope(wallindex).*travel_pos(1)-intercept(wallindex))./(slope(wallindex).*hit_vector(1)-hit_vector(2)));
                    if dist<minobdist
                        minobdist=dist;
                    end
                end
                minobdist=minobdist-2*ballradius;
                % speeds up integration rate
                travel_pos=travel_pos+max(1.0,0.95*minobdist).*hit_vector;
                
            end
        end
        angleindex=angleindex+1;
    end
    anglediffmin=99999;
    pockdistmin=99999;
    numangleindex=angleindex-1;
    %  looks at all angles of impact for which the impacted ball landed in
    %  a pocket, and culls out those with smallest angle from center of
    %  ball, and smallest ball-to-pocket distance
    for angleindex=1:numangleindex
        if success_flag(angleindex)==1
            anglediff=abs(hit_angle(angleindex)-angle_cue_stripe(object));
            if anglediff<anglediffmin
                anglediffmin=anglediff;
                pocket_object(object)=pocket_num(angleindex);
                hit_pointb_stripe(object,:)=hit_pointb(angleindex,:);
                hit_point_stripe(object,:)=hit_point(angleindex,:);
                hit_point2b_stripe(object,:)=hit_point2b(angleindex,:);
                hit_point2_stripe(object,:)=hit_point2(angleindex,:);
            end
            if pocket_dist(angleindex)+pocket_dist2(angleindex)<pockdistmin
                pockdistmin=pocket_dist(angleindex);
                pocket_object2(object)=pocket_num(angleindex);
                hit_pointb_stripe2(object,:)=hit_pointb(angleindex,:);
                hit_point_stripe2(object,:)=hit_point(angleindex,:);
                hit_point2b_stripe2(object,:)=hit_point2b(angleindex,:);
                hit_point2_stripe2(object,:)=hit_point2(angleindex,:);
            end
        end
    end
    if sum(success_flag)~=0
        % plots cue-to-ball-to-pocket lines to superimpose on marked
        % picture; if bounces occurred along the way, inserts bounce
        % locations for lines
        linestripe(1,1)=centroid_cue(1,1);
        linestripe(1,2)=centroid_cue(1,2);
        linestripe(2,1)=hit_pointb_stripe(object,1);
        linestripe(2,2)=hit_pointb_stripe(object,2);
        hold on;
        if dottedline==1
            plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',2.5,'Color','cyan');
        else
            plot(linestripe(:,1),linestripe(:,2),'LineWidth',2.5,'Color','cyan');
        end
        linestripe(2,1)=hit_pointb_stripe2(object,1);
        linestripe(2,2)=hit_pointb_stripe2(object,2);
        hold on;
        if dottedline==1
            plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',2.5,'Color','cyan');
        else
            plot(linestripe(:,1),linestripe(:,2),'LineWidth',2.5,'Color','cyan');
        end
        
        tmpflr=floor((object-1)./NumberStripes);
        if tmpflr==1
            [hit_point_stripe(object,:)] = unreflections(hit_point_stripe(object,:),rotation_angle_top,rotation_point_top,0,1,ballradius);
            [hit_point_stripe2(object,:)] = unreflections(hit_point_stripe2(object,:),rotation_angle_top,rotation_point_top,0,1,ballradius);
        elseif tmpflr==2
            [hit_point_stripe(object,:)] = unreflections(hit_point_stripe(object,:),rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
            [hit_point_stripe2(object,:)] = unreflections(hit_point_stripe2(object,:),rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
        elseif tmpflr==3
            [hit_point_stripe(object,:)] = unreflections(hit_point_stripe(object,:),rotation_angle_left,rotation_point_left,1,1,ballradius);
            [hit_point_stripe2(object,:)] = unreflections(hit_point_stripe2(object,:),rotation_angle_left,rotation_point_left,1,1,ballradius);
        elseif tmpflr==4
            [hit_point_stripe(object,:)] = unreflections(hit_point_stripe(object,:),rotation_angle_right,rotation_point_right,1,0,ballradius);
            [hit_point_stripe2(object,:)] = unreflections(hit_point_stripe2(object,:),rotation_angle_right,rotation_point_right,1,0,ballradius);
        end
        linestripe(1,1)=hit_pointb_stripe(object,1);
        linestripe(1,2)=hit_pointb_stripe(object,2);
        linestripe(2,1)=hit_point_stripe(object,1);
        linestripe(2,2)=hit_point_stripe(object,2);
        hold on;
        if dottedline==1
            plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
        else
            plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
        end
        linestripe(1,1)=hit_pointb_stripe2(object,1);
        linestripe(1,2)=hit_pointb_stripe2(object,2);
        linestripe(2,1)=hit_point_stripe2(object,1);
        linestripe(2,2)=hit_point_stripe2(object,2);
        hold on;
        if dottedline==1
            plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
        else
            plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
        end
        
        
        
        
        tmp=rem(object,NumberStripes);
        if tmp==0
            tmp=NumberStripes;
        end
        tmpflr2=floor((pocket_object(object)-1)./NumberPockets);
        sameflag=0;
        if hit_point2b_stripe(object,1)==hit_point2_stripe(object,1)
            sameflag=1;
        end
        if tmpflr2==1
            [hit_point2_stripe(object,:)] = unreflections(hit_point2_stripe(object,:),rotation_angle_top,rotation_point_top,0,1,ballradius);
        elseif tmpflr2==2
            [hit_point2_stripe(object,:)] = unreflections(hit_point2_stripe(object,:),rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
        elseif tmpflr2==3
            [hit_point2_stripe(object,:)] = unreflections(hit_point2_stripe(object,:),rotation_angle_left,rotation_point_left,1,1,ballradius);
        elseif tmpflr2==4
            [hit_point2_stripe(object,:)] = unreflections(hit_point2_stripe(object,:),rotation_angle_right,rotation_point_right,1,0,ballradius);
        end
        if sameflag==1
            hit_point2b_stripe(object,:)=hit_point2_stripe(object,:);
        end
        sameflag=0;
        if hit_point2b_stripe2(object,1)==hit_point2_stripe2(object,1)
            sameflag=1;
        end
        tmpflr3=floor((pocket_object2(object)-1)./NumberPockets);
        if tmpflr3==1
            [hit_point2_stripe2(object,:)] = unreflections(hit_point2_stripe2(object,:),rotation_angle_top,rotation_point_top,0,1,ballradius);
        elseif tmpflr3==2
            [hit_point2_stripe2(object,:)] = unreflections(hit_point2_stripe2(object,:),rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
        elseif tmpflr3==3
            [hit_point2_stripe2(object,:)] = unreflections(hit_point2_stripe2(object,:),rotation_angle_left,rotation_point_left,1,1,ballradius);
        elseif tmpflr3==4
            [hit_point2_stripe2(object,:)] = unreflections(hit_point2_stripe2(object,:),rotation_angle_right,rotation_point_right,1,0,ballradius);
        end
        if sameflag==1
            hit_point2b_stripe2(object,:)=hit_point2_stripe2(object,:);
        end
        
        
        if tmpflr==0
            linestripe(1,1)=centroid_stripe(tmp,1);
            linestripe(1,2)=centroid_stripe(tmp,2);
            linestripe(2,1)=hit_point2b_stripe(object,1);
            linestripe(2,2)=hit_point2b_stripe(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
            linestripe(1,1)=hit_point2b_stripe(object,1);
            linestripe(1,2)=hit_point2b_stripe(object,2);
            linestripe(2,1)=hit_point2_stripe(object,1);
            linestripe(2,2)=hit_point2_stripe(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
        else
            linestripe(1,1)=centroid_stripe(tmp,1);
            linestripe(1,2)=centroid_stripe(tmp,2);
            linestripe(2,1)=hit_point2_stripe(object,1);
            linestripe(2,2)=hit_point2_stripe(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
        end
        
        if tmpflr==0
            linestripe(1,1)=centroid_stripe(tmp,1);
            linestripe(1,2)=centroid_stripe(tmp,2);
            linestripe(2,1)=hit_point2b_stripe2(object,1);
            linestripe(2,2)=hit_point2b_stripe2(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
            linestripe(1,1)=hit_point2b_stripe2(object,1);
            linestripe(1,2)=hit_point2b_stripe2(object,2);
            linestripe(2,1)=hit_point2_stripe2(object,1);
            linestripe(2,2)=hit_point2_stripe2(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
        else
            
            linestripe(1,1)=centroid_stripe(tmp,1);
            linestripe(1,2)=centroid_stripe(tmp,2);
            linestripe(2,1)=hit_point2_stripe2(object,1);
            linestripe(2,2)=hit_point2_stripe2(object,2);
            hold on;
            if dottedline==1
                plot(linestripe(:,1),linestripe(:,2),'--c','LineWidth',0.5,'Color','cyan');
            else
                plot(linestripe(:,1),linestripe(:,2),'LineWidth',0.5,'Color','cyan');
            end
        end
    end
end
junk=1;
end


function v = homography_solve(pin, pout)
% HOMOGRAPHY_SOLVE finds a homography from point pairs
%   V = HOMOGRAPHY_SOLVE(PIN, POUT) takes a 2xN matrix of input vectors and
%   a 2xN matrix of output vectors, and returns the homogeneous
%   transformation matrix that maps the inputs to the outputs, to some
%   approximation if there is noise.
%
%   This uses the SVD method of
%   http://www.robots.ox.ac.uk/%7Evgg/presentations/bmvc97/criminispaper/node3.html

% David Young, University of Sussex, February 2008

if ~isequal(size(pin), size(pout))
    error('Points matrices different sizes');
end
if size(pin, 1) ~= 2
    error('Points matrices must have two rows');
end
n = size(pin, 2);
if n < 4
    error('Need at least 4 matching points');
end

% Solve equations using SVD
x = pout(1, :); y = pout(2,:); X = pin(1,:); Y = pin(2,:);
rows0 = zeros(3, n);
rowsXY = -[X; Y; ones(1,n)];
hx = [rowsXY; rows0; x.*X; x.*Y; x];
hy = [rows0; rowsXY; y.*X; y.*Y; y];
h = [hx hy];
if n == 4
    [U, ~, ~] = svd(h);
else
    [U, ~, ~] = svd(h, 'econ');
end
v = (reshape(U(:,9), 3, 3)).';
end


function y = homography_transform(x, v)
% HOMOGRAPHY_TRANSFORM applies homographic transform to vectors
%   Y = HOMOGRAPHY_TRANSFORM(X, V) takes a 2xN matrix, each column of which
%   gives the position of a point in a plane. It returns a 2xN matrix whose
%   columns are the input vectors transformed according to the homography
%   V, represented as a 3x3 homogeneous matrix.

q = v * [x; ones(1, size(x,2))];
p = q(3,:);
y = [q(1,:)./p; q(2,:)./p];
end


function [picture_delta,picture_filter,cent,bound,picture_mag]=binarize_image(picture_double,location_r,location_g,location_b,location_m,otsu_flag,plot_binarize,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right)
% binarizes picture by taking inner product of weighting vector (simply the
% table RGB color component values) with each pixel, and binarizing based
% on whether > 0.98;  to locate green balls, takes addtional step of using
% Otsu's method on the gray-scale histogram to discriminate table
% (foreground) from other objects
height=size(picture_double,1);
width=size(picture_double,2);

dotprod=location_r.*picture_r_unit+location_g.*picture_g_unit+location_b.*picture_b_unit;
picture_filter=dotprod>0.98;

if otsu_flag==1
    % Uses Otsu's method to find threshold between fore and background
    rowstart=corner_upper_left(2);
    rowend=corner_lower_left(2);
    colstart=corner_upper_left(1);
    colend=corner_upper_right(1);
    
    Threshold_m=graythresh(picture_mag(rowstart:rowend,colstart:colend));
    
    if location_m>Threshold_m
        picture_filter=picture_filter & picture_mag>=Threshold_m;
    else
        picture_filter=picture_filter & picture_mag<=Threshold_m;
    end
end

% after binarization, uses region labeling, then largest region, to isolate
% table
picture_label=bwlabel(picture_filter,4);
STATS=regionprops(picture_label,'Area','Centroid','BoundingBox');
area=cat(1,STATS.Area);
centroid=cat(1,STATS.Centroid);
[carea,iarea]=max(area);
boundingbox=cat(1,STATS.BoundingBox);

% finds centroid and bounding box of table region
cent=centroid(iarea,:);
bound=boundingbox(iarea,:,:,:,:);

% then creates binarized picture with table (and "holes" from pockets and
% balls) alone
picture_final=picture_filter & (picture_label == iarea);

% creates a filled rectangle binarized image with corner taken from the
% table region bounding box
for column=1:width
    for row=1:height
        if abs(row-cent(2))<bound(4)/2 && abs(column-cent(1))<bound(3)/2
            picture_fill(row,column)=1;
        else
            picture_fill(row,column)=0;
        end
    end
end

% XORs filled rectangle with binarized table, to (hopefully) leave pockets
% and balls while removing as much table and other objects
picture_delta=xor(picture_fill,picture_final);
if otsu_flag==1
    picture_delta=imfill(picture_delta,'holes');
end

if plot_binarize==1
    figure;
    imshow(picture_delta);
end

end


function[pocket,centroid_pocket,boundingbox_pocket,pocketradius,picture_pockets,NumberPockets]=find_pockets(SE,picture_label,cent,bound,area,centroid,boundingbox,SEradius,plot_pocket)
% locates positions of pockets by finding objects closest to the vertices
% of the table bounding box given by region labeling
vertex(1,1)=cent(1)-bound(3)/2;
vertex(1,2)=cent(2)-bound(4)/2;
vertex(2,1)=cent(1);
vertex(2,2)=cent(2)-bound(4)/2;
vertex(3,1)=cent(1)+bound(3)/2;
vertex(3,2)=cent(2)-bound(4)/2;
vertex(4,1)=cent(1)-bound(3)/2;
vertex(4,2)=cent(2)+bound(4)/2;
vertex(5,1)=cent(1);
vertex(5,2)=cent(2)+bound(4)/2;
vertex(6,1)=cent(1)+bound(3)/2;
vertex(6,2)=cent(2)+bound(4)/2;
for vert=1:6
    min(vert)=9999;
end
for vert=1:6
    for object=1:length(area)
        vertexdist(vert)=sqrt((centroid(object,1)-vertex(vert,1)).^2+(centroid(object,2)-vertex(vert,2)).^2);
        if vertexdist(vert)<min(vert) && vertexdist(vert)<8*SEradius
            min(vert)=vertexdist(vert);
            pocket(vert)=object;
            centroid_pocket(vert,:)=centroid(object,:);
            boundingbox_pocket(vert,:)=boundingbox(object,:);
        end
    end
end

% finds "radius" of pockets by adding eroded objects bounding box widths to
% the erosion structuring element's radius
radius=(boundingbox_pocket(:,3)+boundingbox_pocket(:,4))/4+SEradius;
[n,xout]=hist(radius);

[maxval,maxloc]=max(n);

pocketradius=xout(maxloc);

% creates a binarized picture of pockets alone
picture_pockets=picture_label*0;
for object=1:6
    picture_pockets=picture_pockets | (pocket(object)==picture_label);
end
picture_pockets=imdilate(picture_pockets,SE);

if plot_pocket==1
    figure;
    imshow(picture_pockets);
end

NumberPockets=6;

end


function[SEradius,SE,picture_label,area,centroid,boundingbox,picture_dilate]=erode_objects(bound,picture_delta,plot_erode,erode_degree)
% erodes all table objects with a disc-shaped structuring element of around
% half the radius of a typical pool ball (as given by ratio of ball to
% table width); erodes the objects to reduce noise and connections between
% balls and each other and pockets, for better object identification
if bound(3)<bound(4)
    SEradius=round(bound(3)./erode_degree);
else
    SEradius=round(bound(4)./erode_degree);
end
SE = strel('disk', SEradius);

picture_erode=imerode(picture_delta,SE);

picture_label=bwlabel(picture_erode,4);
STATS=regionprops(picture_label,'Area','Centroid','BoundingBox');
area=cat(1,STATS.Area);
centroid=cat(1,STATS.Centroid);
boundingbox=cat(1,STATS.BoundingBox);
picture_dilate=imdilate(picture_erode,SE);

if plot_erode==1
    figure;
    imshow(picture_erode);
end

end



function [NumberBalls,ball, centroid_ball,boundingbox_ball,ballradius]=find_balls(pocketradius,centroid_pocket,picture_delta,picture_double,area,pocket,centroid,boundingbox,SEradius,bright_thresh)
% identifies balls by removing identified pockets as potential candidates,
% then removing objects from consideration which do not exhibit "shiny"
% properties (histogram peaks at the max magnitude); then creates a
% binarized template of the average ball object, and AND convolves with
% each binarized ball image to get better estimate of ball center coords
index=1;
% removes pockets from consideration
for object=1:length(area)
    distmin=99999;
    for pocketindex=1:6
        dist=sqrt((centroid_pocket(pocketindex,1)-centroid(object,1)).^2+(centroid_pocket(pocketindex,2)-centroid(object,2)).^2);
        if dist<distmin
            distmin=dist;
        end
    end
    if distmin>0.8*pocketradius
        ball(index)=object;
        centroid_ball(index,:)=centroid(object,:);
        boundingbox_ball(index,:)=boundingbox(object,:);
        index=index+1;
    end
end
NumberBalls=length(ball);

radius=(boundingbox_ball(:,3)+boundingbox_ball(:,4))/4+SEradius;
[n,xout]=hist(radius);

[maxval,maxloc]=max(n);

ballradius=xout(maxloc);

NumberBallsTmp=NumberBalls;
balltmp=ball;
centroid_balltmp=centroid_ball;
boundingbox_balltmp=boundingbox_ball;
% removes non-shiny objects from consideration
for object=1:NumberBalls
    rowstart=round(centroid_ball((object),2)-ballradius);
    rowend=round(centroid_ball((object),2)+ballradius);
    colstart=round(centroid_ball((object),1)-ballradius);
    colend=round(centroid_ball((object),1)+ballradius);
    pic_ball=picture_double(rowstart:rowend,colstart:colend,:);
    pic_ball_m=sqrt(pic_ball(:,:,1).^2+pic_ball(:,:,2).^2+pic_ball(:,:,3).^2);
    [hm,xoutm]=imhist(pic_ball_m);
    [maxval,maxloc]=max(hm);
    if maxloc/length(hm)<bright_thresh
        for i=object:NumberBalls-1
            balltmp(i)=balltmp(i+1);
            centroid_balltmp(i,:)=centroid_balltmp(i+1,:);
            boundingbox_balltmp(i,:)=boundingbox_balltmp(i+1,:);
        end
        NumberBallsTmp=NumberBallsTmp-1;
    end
end
NumberBalls=NumberBallsTmp;
ball=balltmp;
centroid_ball=centroid_balltmp;
boundingbox_ball=boundingbox_balltmp;

% creates average-ball binarized template
firsttime=0;
for object=1:NumberBalls
    rowstart=round(centroid_ball((object),2)-1.0*ballradius);
    rowend=rowstart+round(2*1.0*ballradius);
    colstart=round(centroid_ball((object),1)-1.0*ballradius);
    colend=colstart+round(2*1.0*ballradius);
    if firsttime==0
        pic_ball_bin=picture_delta(rowstart:rowend,colstart:colend);
        firsttime=1;
    else
        pic_ball_bin=pic_ball_bin+picture_delta(rowstart:rowend,colstart:colend);
    end
end
pic_ball_bin=round(pic_ball_bin./NumberBalls);

% uses AND convolution of template with binarized ball image to identify
% the most likely center (x,y coords) of each ball
extent=round(ballradius/8);
for object=1:NumberBalls
    pic_ball_max=0;
    rowoff=0;
    coloff=0;
    for i=-extent:extent
        for j=-extent:extent
            rowstart=round(centroid_ball((object),2)-1.0*ballradius+i);
            rowend=rowstart+round(2*1.0*ballradius);
            colstart=round(centroid_ball((object),1)-1.0*ballradius+j);
            colend=colstart+round(2*1.0*ballradius);
            pic_tmp=picture_delta(rowstart:rowend,colstart:colend);
            pic_ball_sum=sum(sum(pic_ball_bin & pic_tmp));
            if pic_ball_sum>pic_ball_max
                pic_ball_max=pic_ball_sum;
                rowoff=i;
                coloff=j;
            end
        end
    end
    centroid_ball(object,2)=centroid_ball(object,2)+rowoff;
    centroid_ball(object,1)=centroid_ball(object,1)+coloff;
end
end


function [ball,centroid_ball,boundingbox_ball,NumberBalls]=merge_greens(NumberBalls,NumberBalls2,ball,centroid_ball,boundingbox_ball,ball2,centroid_ball2,boundingbox_ball2,ballradius)
% finds which ball objects obtained using Otsu thresholding do not appear
% in the ball objects obtained without Otsu (usually the green balls) and
% appends these balls to the ball list
balldup=zeros(NumberBalls2,1);
for i=1:NumberBalls
    for j=1:NumberBalls2
        centdist=sqrt((centroid_ball(i,1)-centroid_ball2(j,1)).^2+(centroid_ball(i,2)-centroid_ball2(j,2)).^2);
        if centdist<1.8*ballradius
            balldup(j)=1;
        end
    end
end
index=1;
for j=1:NumberBalls2
    if balldup(j)==0
        ball(NumberBalls+index)=ball2(j);
        centroid_ball(NumberBalls+index,:)=centroid_ball2(j,:);
        boundingbox_ball(NumberBalls+index,:)=boundingbox_ball2(j,:);
        index=index+1;
    end
end
NumberBalls=size(centroid_ball,1);
end


function [cue_ball,centroid_cue,multmat,NumberEight,eight_ball,centroid_eight]=find_cue_eight(NumberBalls,centroid_ball,ballradius,picture_delta,picture_double)
wmax=0;
wmin=9999999;

index=1;
% finds which of the ball objects is the cue ball, and which is the eight
% ball, by analyzing the histogram and determining which ball has
% histogram values most dominantly in the white region (around 0.57 in each R, G
% and B component); this is determined to be the cue ball;  looks at which
% ball is darkest (has lowest peak gray-scale value) and labels this the
% eight ball; from cue ball, gets pixel R, G and B multipliers which white
% balance the cue ball, to be used to white balance all other balls
for object=1:NumberBalls
    rowstart=round(centroid_ball((object),2)-1.0*ballradius);
    rowend=rowstart+round(2*1.0*ballradius);
    colstart=round(centroid_ball((object),1)-1.0*ballradius);
    colend=colstart+round(2*1.0*ballradius);
    pic_ball_bin=picture_delta(rowstart:rowend,colstart:colend);
    pic_ball=picture_double(rowstart:rowend,colstart:colend,:);
    pic_ball_m=sqrt(pic_ball(:,:,1).^2+pic_ball(:,:,2).^2+pic_ball(:,:,3).^2);
    pic_ball_r=pic_ball(:,:,1)./pic_ball_m.*pic_ball_bin;
    pic_ball_g=pic_ball(:,:,2)./pic_ball_m.*pic_ball_bin;
    pic_ball_b=pic_ball(:,:,3)./pic_ball_m.*pic_ball_bin;
    pic_ball_m=pic_ball_m.*pic_ball_bin;
    [hr,xoutr]=imhist(pic_ball_r);
    [hg,xoutg]=imhist(pic_ball_g);
    [hb,xoutb]=imhist(pic_ball_b);
    [hm,xoutm]=imhist(pic_ball_m);
    hr(1)=0;
    hg(1)=0;
    hb(1)=0;
    hm(1)=0;
    
    scale=length(hr)/2;
    startpos=round(scale./xoutr(scale)*0.5);
    endpos=round(scale./xoutr(scale)*0.7);
    startpos2=round(0.9*length(hr));
    endpos2=length(hr);
    
    % integrates histogam values around white values
    rtop=sum(hr(startpos:endpos));
    gtop=sum(hg(startpos:endpos));
    btop=sum(hb(startpos:endpos));
    mtop=sum(hm(startpos2:endpos2));
    
    % for integrated white-region histogram values, selects ball with
    % largest integrates values as cue ball; obtains white balance
    % multipliers from cue ball pixel color values
    if (rtop+gtop+btop+mtop)>wmax
        wmax=(rtop+gtop+btop+mtop);
        cue_ball=object;
        centroid_cue(1,:)=centroid_ball(object,:);
        [rows,cols]=size(pic_ball_r);
        for row=1:rows
            for col=1:cols
                tmp=max(pic_ball_r(row,col),pic_ball_g(row,col));
                tmp=max(tmp,pic_ball_b(row,col));
                if tmp~=0
                    multmat(row,col,1)=tmp./pic_ball_r(row,col);
                    multmat(row,col,2)=tmp./pic_ball_g(row,col);
                    multmat(row,col,3)=tmp./pic_ball_b(row,col);
                else
                    multmat(row,col,1)=1;
                    multmat(row,col,2)=1;
                    multmat(row,col,3)=1;
                end
            end
        end
    end
    
    rbot=sum(hr(startpos:endpos));
    gbot=sum(hg(startpos:endpos));
    bbot=sum(hb(startpos:endpos));
    mbot=sum(hm(startpos2:endpos2));
    
    
    
    % selects darkest object as eight ball
    if mbot<wmin && cue_ball~=object && (rtop+gtop+btop)./(sum(hr)+sum(hg)+sum(hb))>0.3
        wmin=mbot;
        eight_ball=object;
        centroid_eight(1,:)=centroid_ball(object,:);
    end
end
NumberEight=1;

end


function [picture_marked,solid_ball,centroid_solid,stripe_ball,centroid_stripe,NumberSolids,NumberStripes]=find_stripes_solids(cue_ball,eight_ball,multmat,NumberBalls,centroid_ball,ballradius,picture_delta,picture_double,picture_marked)
% differentiates solids from stripes by first white-balancing each ball,
% then taking the inner product of a "white" weighting vector with the ball
% image, then integragin the binarized regions which pass the threshold;
% this integrated value is divided by ball area to obtain percent white
% values; those balls exceeding white percent threshold are labeled
% stripes, those not are labeled solids
solid_index=1;
stripe_index=1;
index=1;
for object=1:NumberBalls
    rowstart=round(centroid_ball((object),2)-1.0*ballradius);
    rowend=rowstart+round(2*1.0*ballradius);
    colstart=round(centroid_ball((object),1)-1.0*ballradius);
    colend=colstart+round(2*1.0*ballradius);
    pic_ball_bin=picture_delta(rowstart:rowend,colstart:colend);
    pic_ball=picture_double(rowstart:rowend,colstart:colend,:);
    % white balances each ball
    pic_ball=pic_ball.*multmat;
    % inserts white-balanced ball image into marked picture
    picture_marked(rowstart:rowend,colstart:colend,:)=pic_ball;
    pic_ball_m=sqrt(pic_ball(:,:,1).^2+pic_ball(:,:,2).^2+pic_ball(:,:,3).^2);
    pic_ball_r=pic_ball(:,:,1)./pic_ball_m.*pic_ball_bin;
    pic_ball_g=pic_ball(:,:,2)./pic_ball_m.*pic_ball_bin;
    pic_ball_b=pic_ball(:,:,3)./pic_ball_m.*pic_ball_bin;
    pic_ball_m=pic_ball_m;
    [hr,xoutr]=imhist(pic_ball_r);
    [hg,xoutg]=imhist(pic_ball_g);
    [hb,xoutb]=imhist(pic_ball_b);
    [hm,xoutm]=imhist(pic_ball_m);
    hr(1)=0;
    hg(1)=0;
    hb(1)=0;
    hm(1)=0;
    
    % inner product with white weight vector
    dotprod=0.57*pic_ball_r+0.57*pic_ball_g+0.57*pic_ball_b;
    pic_ball_tmp=dotprod>0.96 & pic_ball_m>0.9;
    
    [rows,cols]=size(pic_ball);
    
    % those >35% white are labeled stripes, otherwise solids
    white_fraction=sum(sum(pic_ball_tmp))./(pi*ballradius.^2);
    if white_fraction<0.35 && object~= cue_ball && object~= eight_ball
        solid_ball(solid_index)=object;
        centroid_solid(solid_index,:)=centroid_ball(object,:);
        solid_index=solid_index+1;
    elseif object~= cue_ball && object~= eight_ball
        stripe_ball(stripe_index)=object;
        centroid_stripe(stripe_index,:)=centroid_ball(object,:);
        stripe_index=stripe_index+1;
    end
    
end
NumberSolids=solid_index-1;
NumberStripes=stripe_index-1;

end



function [picture_marked]=mark_image(picture_marked,ballradius,NumberSolids,centroid_solid,NumberStripes,centroid_stripe,centroid_cue,centroid_eight,centroid_pocket)
% creates colored rings around balls, corresponding to ball radius and
% centroid; broken rings mark stripes, solid rings mark solids
pixsize=round(3*ballradius/40);
pixsize2=round(10*ballradius/40);
for i=1:NumberSolids
    for theta=0:360
        col=round(centroid_solid(i,1)+ballradius.*cos(theta*pi/180));
        row=round(centroid_solid(i,2)+ballradius.*sin(theta*pi/180));
        picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,1)=0.6;
        picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,2)=1;
        picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,3)=1;
    end
end

for i=1:NumberStripes
    for theta=0:360
        if abs(cos(theta*pi/180))>0.9 || abs(sin(theta*pi/180))>0.9
            col=round(centroid_stripe(i,1)+ballradius.*cos(theta*pi/180));
            row=round(centroid_stripe(i,2)+ballradius.*sin(theta*pi/180));
            picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,1)=0.6;
            picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,2)=1;
            picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,3)=1;
        end
    end
end

% creates blue mark at center of cue, and ring around cue
row=centroid_cue(1,2);
col=centroid_cue(1,1);
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,1)=0;
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,2)=0;
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,3)=1;
for theta=0:360
    col=round(centroid_cue(1,1)+ballradius.*cos(theta*pi/180));
    row=round(centroid_cue(1,2)+ballradius.*sin(theta*pi/180));
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,1)=0;
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,2)=0;
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,3)=1;
end

% creates red mark at center of eight, and ring around eight
row=centroid_eight(1,2);
col=centroid_eight(1,1);
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,1)=1;
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,2)=0;
picture_marked(row-pixsize2:row+pixsize2,col-pixsize2:col+pixsize2,3)=0;
for theta=0:360
    col=round(centroid_eight(1,1)+ballradius.*cos(theta*pi/180));
    row=round(centroid_eight(1,2)+ballradius.*sin(theta*pi/180));
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,1)=1;
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,2)=0;
    picture_marked(row-pixsize:row+pixsize,col-pixsize:col+pixsize,3)=0;
end

% marks center of each pocket
for object=1:6
    row=centroid_pocket(object,2);
    col=centroid_pocket(object,1);
    picture_marked(row-2*pixsize2:row+2*pixsize2,col-2*pixsize2:col+2*pixsize2,1)=0.6;
    picture_marked(row-2*pixsize2:row+2*pixsize2,col-2*pixsize2:col+2*pixsize2,2)=1;
    picture_marked(row-2*pixsize2:row+2*pixsize2,col-2*pixsize2:col+2*pixsize2,3)=1;
end
figure;
imshow(picture_marked);

end


function [location_r,location_g,location_b,location_m]=find_table_color(height,width,picture_mag,picture_r_unit,picture_g_unit,picture_b_unit)
% from center 1/4 of image, obtains R,G and B histograms, then locates
% peaks in each histogram and assigns as table color; uses these values as
% weight vector in subsequent binarization
rowstart=height*1/4;
rowend=height*3/4;
colstart=width*1/4;
colend=width*3/4;


picture_bbox_mag=picture_mag(rowstart:rowend,colstart:colend);
picture_bbox_r_unit=picture_r_unit(rowstart:rowend,colstart:colend);
picture_bbox_g_unit=picture_g_unit(rowstart:rowend,colstart:colend);
picture_bbox_b_unit=picture_b_unit(rowstart:rowend,colstart:colend);

% finds histogram characteristics for each component
[nr,xoutr]=imhist(picture_bbox_r_unit);
[ng,xoutg]=imhist(picture_bbox_g_unit);
[nb,xoutb]=imhist(picture_bbox_b_unit);
[nm,xoutm]=imhist(picture_bbox_mag);

% finds location of peak values in component histograms; this will
% indicate table foreground in later Otsu test
[cr,ir]=max(nr);
[cg,ig]=max(ng);
[cb,ib]=max(nb);
[cm,im]=max(nm);

location_r=xoutr(ir);
location_g=xoutr(ig);
location_b=xoutr(ib);
location_m=xoutr(im);
end


function [picture_double]=rectify_image(picture_filter,picture_double,plot_rectified,corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right)
% from outer table edge lines, projects to find 4 outer corners, then
% inputs these along with rectified table corner values into homography
% algorithm; creates transformation matrix and executes on original image
% to produce rectified image (such that is true look-down perspective, with
% top and bottom table edges parallel to x-axis, and left and right
% parallel to y-axis)
% [corner_outer_upper_left]=cornerfind(intercept(1),intercept(5),slope(1),slope(5));
% [corner_outer_upper_right]=cornerfind(intercept(2),intercept(6),slope(2),slope(6));
% [corner_outer_lower_left]=cornerfind(intercept(3),intercept(5),slope(3),slope(5));
% [corner_outer_lower_right]=cornerfind(intercept(4),intercept(6),slope(4),slope(6));

midrow=size(picture_filter,1)./2;
midcol=size(picture_filter,2)./2;
half_width=round(2353/2)*1296/midcol;
half_height=round(1274/2)*1296/midcol;
pout=[midcol-half_width midcol+half_width midcol-half_width midcol+half_width;midrow-half_height midrow-half_height midrow+half_height midrow+half_height];
pin=[corner_upper_left' corner_upper_right' corner_lower_left' corner_lower_right'];

v=homography_solve(pin,pout);

T=maketform('projective',((v))');
picture_homography=imtransform(picture_double,T);

if plot_rectified==1
    figure;
    imshow(picture_homography);
    t = text(20,90,'Rectified to Top-Down View','FontSize',20, 'FontWeight','demi');
end

picture_double=picture_homography;
end


function [NumberStripesTmp,centroid_stripe,NumberSolidsTmp,centroid_solid,NumberEightTmp,centroid_eight,NumberPocketsTmp,centroid_pocket,rotation_angle_top,rotation_point_top,rotation_angle_bottom,rotation_point_bottom,rotation_angle_left,rotation_point_left,rotation_angle_right,rotation_point_right]=find_reflections(ballradius,slope,intercept,NumberStripes,centroid_stripe,NumberSolids,centroid_solid,NumberEight,centroid_eight,NumberPockets,centroid_pocket)
% instead of running simulation and "bouncing" balls off inner walls,
% creates "reflections" of each ball around each wall, and runs sim against
% reflections
% These rotation angles will be used to rotate inner wall lines to
% perfectly vertical or horizontal, which should not be necessary if the
% homographic rectification was done correctly, but is executed anyway just
% in case

% Upward reflections of balls and pockets
rotation_line_top_slope=(slope(7)+slope(8))./2;
rotation_line_top_intercept=(intercept(7)+intercept(8))./2;
rotation_point_top(1)=0;
rotation_point_top(2)=rotation_line_top_intercept;
rotation_angle_top=atan(rotation_line_top_slope);

NumberStripesTmp=NumberStripes;
NumberSolidsTmp=NumberSolids;
NumberEightTmp=NumberEight;
NumberPocketsTmp=NumberPockets;

[centroid_stripe,NumberStripesTmp] = reflections(centroid_stripe,NumberStripes,NumberStripesTmp,rotation_angle_top,rotation_point_top,0,1,ballradius);
[centroid_solid,NumberSolidsTmp] = reflections(centroid_solid,NumberSolids,NumberSolidsTmp,rotation_angle_top,rotation_point_top,0,1,ballradius);
[centroid_eight,NumberEightTmp] = reflections(centroid_eight,NumberEight,NumberEightTmp,rotation_angle_top,rotation_point_top,0,1,ballradius);
[centroid_pocket,NumberPocketsTmp] = reflections(centroid_pocket,NumberPockets,NumberPocketsTmp,rotation_angle_top,rotation_point_top,0,1,ballradius);

% Downward reflections of balls and pockets
rotation_line_bottom_slope=(slope(9)+slope(10))./2;
rotation_line_bottom_intercept=(intercept(9)+intercept(10))./2;
rotation_point_bottom(1)=0;
rotation_point_bottom(2)=rotation_line_bottom_intercept;
rotation_angle_bottom=atan(rotation_line_bottom_slope);

[centroid_stripe,NumberStripesTmp] = reflections(centroid_stripe,NumberStripes,NumberStripesTmp,rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
[centroid_solid,NumberSolidsTmp] = reflections(centroid_solid,NumberSolids,NumberSolidsTmp,rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
[centroid_eight,NumberEightTmp] = reflections(centroid_eight,NumberEight,NumberEightTmp,rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);
[centroid_pocket,NumberPocketsTmp] = reflections(centroid_pocket,NumberPockets,NumberPocketsTmp,rotation_angle_bottom,rotation_point_bottom,0,0,ballradius);

% Leftward reflections of balls and pockets
rotation_line_left_slope=slope(11);
rotation_line_left_intercept=intercept(11);
rotation_point_left(1)=-rotation_line_left_intercept./rotation_line_left_slope;
rotation_point_left(2)=0;
rotation_angle_left=-atan(1./rotation_line_left_slope);

[centroid_stripe,NumberStripesTmp] = reflections(centroid_stripe,NumberStripes,NumberStripesTmp,rotation_angle_left,rotation_point_left,1,1,ballradius);
[centroid_solid,NumberSolidsTmp] = reflections(centroid_solid,NumberSolids,NumberSolidsTmp,rotation_angle_left,rotation_point_left,1,1,ballradius);
[centroid_eight,NumberEightTmp] = reflections(centroid_eight,NumberEight,NumberEightTmp,rotation_angle_left,rotation_point_left,1,1,ballradius);
[centroid_pocket,NumberPocketsTmp] = reflections(centroid_pocket,NumberPockets,NumberPocketsTmp,rotation_angle_left,rotation_point_left,1,1,ballradius);

% Rightward reflections of balls and pockets
rotation_line_right_slope=slope(12);
rotation_line_right_intercept=intercept(12);
rotation_point_right(1)=-rotation_line_right_intercept./rotation_line_right_slope;
rotation_point_right(2)=0;
rotation_angle_right=-atan(1./rotation_line_right_slope);

[centroid_stripe,NumberStripesTmp] = reflections(centroid_stripe,NumberStripes,NumberStripesTmp,rotation_angle_right,rotation_point_right,1,0,ballradius);
[centroid_solid,NumberSolidsTmp] = reflections(centroid_solid,NumberSolids,NumberSolidsTmp,rotation_angle_right,rotation_point_right,1,0,ballradius);
[centroid_eight,NumberEightTmp] = reflections(centroid_eight,NumberEight,NumberEightTmp,rotation_angle_right,rotation_point_right,1,0,ballradius);
[centroid_pocket,NumberPocketsTmp] = reflections(centroid_pocket,NumberPockets,NumberPocketsTmp,rotation_angle_right,rotation_point_right,1,0,ballradius);

end

function [slope,intercept,endp7,endp8,endp9,endp10,endp11,endp12]=find_walls(centroid_pocket,bound,picture_filter,picture_mag,picture_pockets)
% finds each outer table edge and inner wall line by radiating inward from
% edge of picture until running into white edges in binarized table image
ratio=5;
dely=round(bound(4)/6);

buffer=zeros(dely,1);




for wallindex=1:6
    
    if wallindex==1
        usex=0;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(1,1);
        cent2=centroid_pocket(3,1);
    elseif wallindex==2
        usex=0;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(2,1);
        cent2=centroid_pocket(3,1);
    elseif wallindex==3
        usex=0;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(4,1);
        cent2=centroid_pocket(5,1);
    elseif wallindex==4
        usex=0;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(5,1);
        cent2=centroid_pocket(6,1);
    elseif wallindex==5
        usex=1;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(1,2);
        cent2=centroid_pocket(4,2);
    else
        usex=1;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(3,2);
        cent2=centroid_pocket(6,2);
    end
    
    start=round(cent1+1/ratio*(cent2-cent1));
    stop=round(cent1+(ratio-1)/ratio*(cent2-cent1));
    
    picture_zero=picture_mag.*0;
    
    index=1;
    if usex==1
        % finding top or bottom table edges
        for y=start:stop
            if startend==1
                x=size(picture_filter,2);
            else
                x=1;
            end
            % looks for binarized white edges of table
            while (picture_filter(y,x)==0)
                x=x+addmult.*1;
            end
            % saves edges in arrays
            xarray(index)=x;
            yarray(index)=y;
            picture_zero(y,x)=1;
            x=x+addmult.*2;
            % integration of gradient vector
            buffer=buffer+(picture_mag(y,x+addmult:addmult:x+addmult.*dely)-picture_mag(y,x-addmult:addmult:x+addmult.*dely-addmult.*2))';
            index=index+1;
        end
    else
        % finding left or right table edges
        for x=start:stop
            if startend==1
                y=size(picture_filter,1);
            else
                y=1;
            end
            while (picture_filter(y,x)==0)
                y=y+addmult.*1;
            end
            xarray(index)=x;
            yarray(index)=y;
            picture_zero(y,x)=1;
            y=y+addmult.*2;
            buffer=buffer+(picture_mag(y+addmult:addmult:y+addmult.*dely,x)-picture_mag(y-addmult:addmult:y+addmult.*dely-addmult.*2,x));
            index=index+1;
        end
    end
    % Hough transform
    [H,theta,rho]=hough(picture_zero,'Theta',-90:0.5:89.5);
    peaks=houghpeaks(H,1);
    t=theta(peaks(1,2))*pi/180;
    r=rho(peaks(1,1));
    if tan(t)==0
        slope(wallindex)=99999;
    else
        slope(wallindex)=-1./tan(t);
    end
    if sin(t)==0
        intercept(wallindex)=-r*slope(wallindex);
    else
        intercept(wallindex)=r./sin(t);
    end
    
end




% finds line characteristics of inner bounce wall, given offset distance from outer
% edge
buffer2=buffer(5:length(buffer));
[maxval,maxloc]=max((buffer2));
[minval,minloc]=min((buffer2));
intercept_offset=min(maxloc,minloc);



for wallindex=1:6
    wallindex2=wallindex+6;
    
    if wallindex==1
        usex=0;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(1,1);
        cent2=centroid_pocket(3,1);
    elseif wallindex==2
        usex=0;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(2,1);
        cent2=centroid_pocket(3,1);
    elseif wallindex==3
        usex=0;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(4,1);
        cent2=centroid_pocket(5,1);
    elseif wallindex==4
        usex=0;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(5,1);
        cent2=centroid_pocket(6,1);
    elseif wallindex==5
        usex=1;
        startend=0;
        addmult=1;
        cent1=centroid_pocket(1,2);
        cent2=centroid_pocket(4,2);
    else
        usex=1;
        startend=1;
        addmult=-1;
        cent1=centroid_pocket(3,2);
        cent2=centroid_pocket(6,2);
    end
    
    start=round(cent1+1/ratio*(cent2-cent1));
    stop=round(cent1+(ratio-1)/ratio*(cent2-cent1));
    
    if usex==0
        intercept(wallindex2)=intercept(wallindex)+addmult.*intercept_offset;
    else
        intercept(wallindex2)=intercept(wallindex)-addmult.*slope(wallindex).*intercept_offset;
    end
    slope(wallindex2)=slope(wallindex);
    
    
    % traverses length of found inner wall lines until pockets found in
    % binarized pocket image (finds white pixels); creates line start and stop
    % points for each wall
    if usex==1
        yinit=start;
        xinit=(yinit-intercept(wallindex2))./slope(wallindex2);
        y=yinit;
        x=xinit;
        while(picture_pockets(round(y),round(x))==0)
            y=y-1;
            x=(y-intercept(wallindex2))./slope(wallindex2);
        end
        endp(1,1)=x;
        endp(1,2)=y;
        y=yinit;
        x=xinit;
        while(picture_pockets(round(y),round(x))==0)
            y=y+1;
            x=(y-intercept(wallindex2))./slope(wallindex2);
        end
        endp(2,1)=x;
        endp(2,2)=y;
    else
        xinit=start;
        yinit=slope(wallindex2).*xinit+intercept(wallindex2);
        x=xinit;
        y=yinit;
        while(picture_pockets(round(y),round(x))==0)
            x=x-1;
            y=slope(wallindex2).*x+intercept(wallindex2);
        end
        endp(1,1)=x;
        endp(1,2)=y;
        x=xinit;
        y=yinit;
        while(picture_pockets(round(y),round(x))==0)
            x=x+1;
            y=slope(wallindex2).*x+intercept(wallindex2);
        end
        endp(2,1)=x;
        endp(2,2)=y;
    end
    if wallindex==1
        endp7=endp;
    elseif wallindex==2
        endp8=endp;
    elseif wallindex==3
        endp9=endp;
    elseif wallindex==4
        endp10=endp;
    elseif wallindex==5
        endp11=endp;
    else
        endp12=endp;
    end
    
end

end



function [dist_cue_stripe,angle_cue_stripe,angledel_cue_stripe,dist_cue_solid,angle_cue_solid,angledel_cue_solid,dist_cue_eight,angle_cue_eight,angledel_cue_eight]=find_ball_angles(ballradius,centroid_cue,NumberStripesTmp,centroid_stripe,NumberSolidsTmp,centroid_solid,NumberEightTmp,centroid_eight)
% finds cue-to-ball vector and angle values for use in subsequent cue guide
% calculations
for object=1:NumberStripesTmp
    vector_cue_stripe(object,:)=centroid_stripe(object,:)-centroid_cue(1,:);
    dist_cue_stripe(object)=sqrt((vector_cue_stripe(object,1)).^2+(vector_cue_stripe(object,2)).^2);
    angle_cue_stripe(object)=180/pi*atan2(vector_cue_stripe(object,2),vector_cue_stripe(object,1));
    angledel_cue_stripe(object)=180/pi*atan(2*ballradius./dist_cue_stripe(object));
end
for object=1:NumberSolidsTmp
    vector_cue_solid(object,:)=centroid_solid(object,:)-centroid_cue(1,:);
    dist_cue_solid(object)=sqrt((vector_cue_solid(object,1)).^2+(vector_cue_solid(object,2)).^2);
    angle_cue_solid(object)=180/pi*atan2(vector_cue_solid(object,2),vector_cue_solid(object,1));
    angledel_cue_solid(object)=180/pi*atan(2*ballradius./dist_cue_solid(object));
end
for object=1:NumberEightTmp
    vector_cue_eight(object,:)=centroid_eight(object,:)-centroid_cue(1,:);
    dist_cue_eight(object)=sqrt((vector_cue_eight(object,1)).^2+(vector_cue_eight(object,2)).^2);
    angle_cue_eight(object)=180/pi*atan2(vector_cue_eight(object,2),vector_cue_eight(object,1));
    angledel_cue_eight(object)=180/pi*atan(2*ballradius./dist_cue_eight(object));
end
end




function [corner_upper_left,corner_upper_right,corner_lower_left,corner_lower_right] = find_corners(picture_filter)
picture_edges=imfill(picture_filter,'holes');

picture_label=bwlabel(picture_edges,4);
STATS=regionprops(picture_label,'Area','Centroid','BoundingBox');
area=cat(1,STATS.Area);
centroid=cat(1,STATS.Centroid);
boundingbox=cat(1,STATS.BoundingBox);

cent=centroid(1,:);
bound=boundingbox(1,:);

vertex(1,1)=cent(1)-bound(3)/2;
vertex(1,2)=cent(2)-bound(4)/2;
vertex(2,1)=cent(1)+bound(3)/2;
vertex(2,2)=cent(2)-bound(4)/2;
vertex(3,1)=cent(1)-bound(3)/2;
vertex(3,2)=cent(2)+bound(4)/2;
vertex(4,1)=cent(1)+bound(3)/2;
vertex(4,2)=cent(2)+bound(4)/2;


start=round((vertex(3,2)-vertex(1,2))./4+vertex(1,2));
stop=round((vertex(3,2)-vertex(1,2)).*3/4+vertex(1,2));
picture_zero=picture_edges*0;
for y=start:stop
    x=1;
    % looks for binarized white edges of table
    while (picture_edges(y,x)==0)
        x=x+1;
    end
    % saves edges in arrays
    picture_zero(y,x)=1;
end
% Hough transform
[H,theta,rho]=hough(picture_zero,'Theta',-90:0.1:89.999);
peaks=houghpeaks(H,1);
t=theta(peaks(1,2))*pi/180;
r=rho(peaks(1,1));
if tan(t)==0
    slope=99999;
else
    slope=-1./tan(t);
end
if sin(t)==0
    intercept=-r(index)*slope;
else
    intercept=r./sin(t);
end

slope_left=slope;
intercept_left=intercept;


start=round((vertex(4,2)-vertex(2,2))./4+vertex(2,2));
stop=round((vertex(4,2)-vertex(2,2)).*3/4+vertex(2,2));
picture_zero=picture_edges*0;
for y=start:stop
    x=size(picture_edges,2);
    % looks for binarized white edges of table
    while (picture_edges(y,x)==0)
        x=x-1;
    end
    % saves edges in arrays
    picture_zero(y,x)=1;
end
% Hough transform
[H,theta,rho]=hough(picture_zero,'Theta',-90:0.1:89.999);
peaks=houghpeaks(H,1);
t=theta(peaks(1,2))*pi/180;
r=rho(peaks(1,1));
if tan(t)==0
    slope=99999;
else
    slope=-1./tan(t);
end
if sin(t)==0
    intercept=-r(index)*slope;
else
    intercept=r./sin(t);
end

slope_right=slope;
intercept_right=intercept;


start=round((vertex(2,1)-vertex(1,1))./4+vertex(1,1));
stop=round((vertex(2,1)-vertex(1,1)).*3/4+vertex(1,1));
picture_zero=picture_edges*0;
for x=start:stop
    y=1;
    % looks for binarized white edges of table
    while (picture_edges(y,x)==0)
        y=y+1;
    end
    % saves edges in arrays
    picture_zero(y,x)=1;
end
% Hough transform
[H,theta,rho]=hough(picture_zero,'Theta',-90:0.1:89.999);
peaks=houghpeaks(H,1);
t=theta(peaks(1,2))*pi/180;
r=rho(peaks(1,1));
if tan(t)==0
    slope=99999;
else
    slope=-1./tan(t);
end
if sin(t)==0
    intercept=-r(index)*slope;
else
    intercept=r./sin(t);
end

slope_top=slope;
intercept_top=intercept;


start=round((vertex(4,1)-vertex(3,1))./4+vertex(3,1));
stop=round((vertex(4,1)-vertex(3,1)).*3/4+vertex(3,1));
picture_zero=picture_edges*0;
for x=start:stop
    y=size(picture_edges,1);
    % looks for binarized white edges of table
    while (picture_edges(y,x)==0)
        y=y-1;
    end
    % saves edges in arrays
    picture_zero(y,x)=1;
end
% Hough transform
[H,theta,rho]=hough(picture_zero,'Theta',-90:0.1:89.999);
peaks=houghpeaks(H,1);
t=theta(peaks(1,2))*pi/180;
r=rho(peaks(1,1));
if tan(t)==0
    slope=99999;
else
    slope=-1./tan(t);
end
if sin(t)==0
    intercept=-r(index)*slope;
else
    intercept=r./sin(t);
end



slope_bottom=slope;
intercept_bottom=intercept;





corner_upper_left(1)=(intercept_left-intercept_top)./(slope_top-slope_left);
corner_upper_left(2)=slope_top.*corner_upper_left(1)+intercept_top;

corner_upper_right(1)=(intercept_right-intercept_top)./(slope_top-slope_right);
corner_upper_right(2)=slope_top.*corner_upper_right(1)+intercept_top;

corner_lower_left(1)=(intercept_left-intercept_bottom)./(slope_bottom-slope_left);
corner_lower_left(2)=slope_bottom.*corner_lower_left(1)+intercept_bottom;

corner_lower_right(1)=(intercept_right-intercept_bottom)./(slope_bottom-slope_right);
corner_lower_right(2)=slope_bottom.*corner_lower_right(1)+intercept_bottom;

junk=1;
end
%}

